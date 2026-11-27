# Inkling GGUF bounded expert residency

Status: engineering design for STRATA #7169  
Branch: `fix/inkling-gguf-bounded-host-memory`

## Problem

The CUDA GGUF/HF loader computes `safe_host_ram_cap(ram_experts_gb, ssd_experts)`, but its GGUF expert placement currently assigns every expert that does not fit in VRAM to tier 2 (host). It then allocates one host arena of `uh` bytes. The cap is used only to print a warning. Consequently, `--ram-experts-gb` and `--ssd-experts` do not bound the GGUF host allocation.

The `.strata` loader already has a three-tier policy (VRAM, bounded RAM, SSD cold store). The GGUF loader cannot directly reuse its current cold-store descriptors: a GGUF expert body is synthesized from separate gate/up/down tensors, and the GGUF adapter emits a canonical `GGUF_RAW` header plus those tensor payloads.

## Constraints

- Preserve native GGUF expert payload bytes and quantization. Do not dequantize or requantize cold experts.
- Support fused gate/up and separate gate/up layouts, including split GGUF shards.
- Preserve per-expert gate, up and down scales and the exact `dn_offset` / code metadata.
- Bound resident host memory by the same policy as the `.strata` path; the cold tier must not silently fall back to an unbounded arena.
- Keep the existing host CPU execution and GPU prefill staging contracts intact.
- Fail clearly if a model/layout cannot be streamed safely; never silently exceed the configured cap.

## Source layout and scheduler capability

The GGUF adapter exposes a canonical read_expert_body() operation. It reconstructs each expert from its source tensors, writes the 32-byte GGUF_RAW header and preserves the payload bytes. The CUDA loader may then apply its existing GGUF-to-SF4 conversion for formats without a native GPU kernel.

The first cold-tier implementation deliberately uses this adapter boundary rather than reading tensor spans directly:

- The cold descriptor stores a synthetic expert key/offset and the final resident body size.
- The source-reader callback calls InkModelFile::read_expert_body(); if need_sf4_repack is set, it runs the same repack_gguf_raw_body_to_sf4() conversion used by eager loading.
- The callback writes the complete final expert directly into one bounded ring slot. It does not allocate a model-sized host arena or retain a per-expert copy.
- The callback's temporary source and converted vectors are limited to one expert per active reader. The reader count is capped at four for this path.
- Synthetic whole-expert requests are demand-only in this first patch; chunked speculative prefetch is intentionally skipped because splitting a conversion into arbitrary 512 KiB reads would repeat or corrupt the conversion. The existing .strata file-backed path retains its projection-level prefetch behavior.

The scheduler already supports three independent projection spans through StreamExpertReq::make_three_proj(). A later optimization can use that interface for native GGUF_RAW experts that need no repack, reading fused or separate gate/up and down directly from shard mappings. It is not required for correctness of the initial bounded-memory path.

## Proposed implementation

1. **Bounded placement.** Classify each external-format expert as VRAM, resident RAM, or cold using safe_host_ram_cap(). Only resident RAM experts contribute to the host arena size. The cold ring is separately bounded and reported as fixed I/O working memory.
2. **Source-backed cold store.** Extend InkColdStore with a reader callback mode while keeping the existing .strata direct-I/O mode unchanged. Whole-expert GGUF requests use a single scheduler span; the descriptor maps the logical expert ID to its directory entry.
3. **Conversion parity.** Reuse InkModelFile::read_expert_body() and the existing GGUF-to-SF4 conversion path so cold and resident experts have identical internal bytes and format codes.
4. **Hybrid integration.** Cold experts use InkHostExpertMeta::cold_ssd with a null resident body. The existing CPU demand path and CUDA prefill DMA path acquire and release the cold ring slots. Adaptive hot-tier duplication is disabled for external formats so it cannot exceed the requested RAM cap.
5. **Bound supporting allocations.** Report resident arena bytes, cold-ring bytes, and the maximum per-reader conversion scratch. Keep the cold ring capped at 2 GiB (minimum 256 MiB when explicitly sizing it; 512 MiB default), with the existing minimum slot count.
6. **Test incrementally.** Add host-only tests for tier accounting, zero-RAM/SSD-only policy, directory-index versus logical-expert-key mapping, callback failure propagation, ring pin/eviction, concurrent demand, and hard resident-arena budget assertions. Add CUDA build/runtime smoke coverage separately.

## Acceptance criteria

- With a GGUF model, the planned and allocated resident expert arena never exceeds the computed RAM budget (excluding separately reported fixed runtime overhead).
- `--ram-experts-gb 0` and `--ssd-experts` allocate no full-model host expert arena.
- Cold experts reconstruct byte-for-byte identically to `GgufAdapter::read_expert_body()`.
- No slot is evicted while a CPU reader or CUDA DMA consumer still holds it.
- Existing `.strata` residency and cold-store behavior remain unchanged.
- Logs report VRAM/RAM/cold expert counts and exact ring bytes, making memory planning auditable.

## Known non-goals for the first patch

- No change to quantization formats or model conversion.
- No change to launcher family detection or speculative drafting.
- No promise that a particular model will fit the MSI's 4 GiB VRAM; the goal is to prevent unbounded host allocation and enable bounded streaming.

## References

- STRATA issue: https://github.com/Niko1221/Strata/issues/7169
- CUDA runner: `engine/src/inkling2/cuda/ink_runner.cuh`
- GGUF adapter: `engine/src/inkling2/ink_gguf_loader.h`
- Existing cold store: `engine/src/inkling2/ink_cold_store.hpp`
- Existing hybrid dispatch: `engine/src/inkling2/cuda/ink_hybrid.cuh`

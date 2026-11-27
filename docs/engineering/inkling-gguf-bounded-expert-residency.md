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

For each physical layer, `GgufAdapter::layers` records mapped pointers, shard-resolved tensor locations and per-expert strides for gate/up/down tensors. The adapter's `read_expert_body()` constructs a canonical contiguous `GGUF_RAW` body: 32-byte header, gate/up bytes, alignment padding, then down bytes.

The important discovery is that `strata::core::StreamExpertReq` already supports three independent projection spans via `make_three_proj()`. This is sufficient for GGUF without rebuilding every expert into an intermediate vector:

- Fused gate/up layout: span 0 reads the fused gate+up expert payload; span 1 is empty; span 2 reads down.
- Separate gate/up layout: spans 0, 1 and 2 read gate, up and down independently.
- Each span writes directly into its final offset in the bounded ring slot. The 32-byte `GgufExpertHeader` can be written to the slot when it is bound.
- `file_idx` can identify the GGUF shard and `file_off` the byte offset relative to that shard's mapped data section. A reader callback can copy from the mapped shard base and zero-pad alignment at EOF. This also handles split GGUF models without assuming that the tensors share a shard.

The existing `InkColdStore` currently opens one physical `.strata` file and discards `file_idx` in its read callback. It needs a source-reader mode that accepts a callback over (shard index, mapped-data offset, destination, length), while preserving the existing file-backed mode unchanged. The scheduler's per-projection readiness flags then allow gate/up and down reads to retain their current overlap semantics.

## Proposed implementation

1. **Separate placement from materialization.** Extend GGUF expert placement to classify each expert as VRAM, RAM, or cold using `safe_host_ram_cap()`. Track exact aligned byte totals for each tier. Experts beyond the RAM budget must not contribute to `uh` or the host arena allocation.
2. **Generalize the cold-store reader.** Add a source-reader mode to `InkColdStore` (or a narrowly scoped GGUF wrapper) that consumes the existing three projection spans. Store shard index, source offsets, destination offsets, byte counts, and the 32-byte per-expert header in each descriptor. Bind and pin a ring slot before issuing reads; release it only after CPU evaluation or CUDA DMA has completed. Keep the current `.strata` direct-I/O reader behavior unchanged.
3. **Build GGUF descriptors from the adapter.** Expose validated shard data bases and tensor offsets/strides from `GgufAdapter`. For fused and split gate/up layouts, generate the correct spans and header scales. Validate every span against its source shard's mapped data length before starting worker threads.
4. **Integrate the CUDA hybrid path.** Host metadata must distinguish resident RAM bodies from cold-ring bodies. The decode CPU path and prefill ping-pong DMA path must acquire/release cold slots with the same lifetime guarantees as current SSD slots. Preserve the existing `InkExpertDev` layout and `GGUF_RAW` kernel dispatch.
5. **Bound all supporting allocations.** Include ring slots, bounce buffers, and prefetch slots in the documented RAM budget. Do not interpret `--ram-experts-gb` as a soft cap.
6. **Test incrementally.** Add host-only tests for tier accounting, zero-RAM/SSD-only policy, exact three-span reconstruction against `read_expert_body()` for fused and split gate/up, shard boundaries, slot pin/eviction, concurrent demand/prefetch, and hard budget assertions. Add CUDA build/runtime smoke coverage separately.

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

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

## Source layout

For each physical layer, `GgufAdapter::layers` records mapped pointers and per-expert strides for gate/up/down tensors. `read_expert_body()` constructs the canonical contiguous `GGUF_RAW` body on demand: 32-byte header, gate/up bytes, alignment padding, then down bytes. This is the correct byte-preserving primitive for a cold-reader callback.

The existing `InkColdStore` assumes each expert is a contiguous byte range in one file. Its `.strata` descriptor and direct-I/O reader therefore cannot be pointed at a GGUF tensor offset. A GGUF cold tier needs a reader abstraction that reconstructs one expert into a bounded ring slot from its original tensor slices.

## Proposed implementation

1. **Separate placement from materialization.** Extend GGUF expert placement to classify each expert as VRAM, RAM, or cold using `safe_host_ram_cap()`. Track exact aligned byte totals for each tier. Experts beyond the RAM budget must not contribute to `uh` or the host arena allocation.
2. **Add a source-backed cold reader.** Add a bounded cold ring whose reader callback receives an expert key and reconstructs that expert into a ring slot through the GGUF adapter. Use the existing `read_expert_body()` semantics initially, but avoid retaining a vector per expert or a whole-model staging arena. The ring must pin slots while DMA/CPU execution is in flight and must participate in existing demand/prefetch scheduling.
3. **Integrate the CUDA hybrid path.** Host metadata must distinguish resident RAM bodies from cold-ring bodies. The decode CPU path and prefill ping-pong DMA path must acquire/release cold slots with the same lifetime guarantees as current SSD slots. Preserve the existing `InkExpertDev` layout and `GGUF_RAW` kernel dispatch.
4. **Bound all supporting allocations.** Include ring slots, bounce buffers, and prefetch slots in the documented RAM budget. Do not interpret `--ram-experts-gb` as a soft cap.
5. **Test incrementally.** Add host-only tests for tier accounting, zero-RAM/SSD-only policy, exact body reconstruction for fused and split gate/up, shard boundaries, slot pin/eviction, concurrent demand/prefetch, and hard budget assertions. Add CUDA build/runtime smoke coverage separately.

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

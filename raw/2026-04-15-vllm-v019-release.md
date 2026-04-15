---
title: "vLLM v0.18.0 and v0.19.0 Release Notes"
source_url: https://github.com/vllm-project/vllm/releases
collected: 2026-04-15
type: release-notes
---

# vLLM v0.18.0 and v0.19.0 Release Notes

## v0.19.0 (April 3, 2026)
448 commits from 197 contributors (54 new).

### Key Features
- **Gemma 4 support** — day-one support for all four Gemma 4 variants including MoE routing and multimodal inputs; first-ever Day 0 support on Google TPUs
- **Model Runner V2 as default** — MRV2 is now the default execution path; V1 remains for unsupported cases
- **Async scheduling as default** — zero-bubble async scheduling now on by default
- **Zero-bubble async scheduling + speculative decoding** — spec decode now works with the async scheduler
- **ViT Full CUDA Graphs** — vision encoders support full CUDA graph capture, reducing per-request overhead for multimodal inference
- **General CPU KV cache offloading** — pluggable policy interface with block-level preemption (PRs #37160, #37874, #34805, #36642, #37853)
- **Batch API support**
- **NVIDIA B300/GB300 support** with optimized all-reduce
- **Pipeline parallelism CUDA graphs** for MRV2

### Performance Numbers (production benchmarks)
- 4,741 tok/s at 100 concurrent requests
- Fastest TTFT at every concurrency level vs. SGLang and TensorRT-LLM
- 1.7x throughput over V0 engine (with MRV2 + async scheduling)
- Prefix caching overhead: <1% at 0% hit rate (effectively free to enable)

---

## v0.18.1 (March 31, 2026)
Patch release.
- Reverted SM100 MLA prefill backend defaults
- Fixed mock.patch resolution for Python ≤3.10
- Disabled monolithic TRTLLM MoE for certain routing types
- Resolved DeepGemm accuracy problems on Blackwell

---

## v0.18.0 (March 20, 2026)
445 commits from 213 contributors (61 new).

### Key Features
- **gRPC Serving Support** — new `--grpc` flag for gRPC-based serving
- **GPU-less Render Serving** — `vllm launch render` separates multimodal preprocessing from GPU inference
- **NGram speculative decoding on GPU** — compatible with async scheduler, significantly reduces spec decode overhead
- **Smart CPU KV offloading** — stores only frequently-reused KV cache blocks; FlexKV as new offloading backend
- **FlashInfer 0.2.6 dependency update**
- **Responses API Streaming Tool Calls** support
- **Ray removed as default dependency**

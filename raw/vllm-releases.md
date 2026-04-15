# vLLM Releases (as of April 2026)
- Source: https://github.com/vllm-project/vllm/releases
- Ingested: 2026-04-14
- Type: GitHub releases

Key releases:
- v0.19.0 (April 2026): current latest
  - Zero-bubble async scheduling + speculative decoding (PR #32951)
  - MRV2: piecewise CUDA graphs for PP (#35162), spec decode rejection sampler (#37238, #37237)
  - ViT full CUDA graphs (#35963)
  - General CPU KV cache offloading (#37160, #37874, #34805)
  - DBO generalization for all models
  - Gemma 4 support (Day 0 on TPUs)
- v0.15.1 (Feb 2026): full NVIDIA Blackwell SM120 support, H200 optimizations
- v0.8.0: V1 engine became default (1.7x throughput over V0)

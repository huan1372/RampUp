# vLLM Benchmarks 2026
- Source: https://www.morphllm.com/vllm-benchmarks
- Ingested: 2026-04-14
- Type: Blog / benchmark report

Key data:
- Clarifai benchmark: GPT-OSS-120B on 2x H100, concurrency 1-100
  - vLLM: 4,741 tok/s at 100 concurrent, fastest TTFT at every level
  - SGLang: most stable ITL (4-21ms), strong at 50 concurrent
  - TensorRT-LLM: best single-request throughput, worst scaling, slowest TTFT
- Interactive targets: TTFT < 200ms, ITL < 30ms on H100 for models up to 70B
- V1 engine: 1.7x throughput over V0, prefix caching <1% overhead at 0% hit
- FP8 on H100/B200: 2x memory reduction, up to 1.6x throughput, minimal accuracy loss
- vLLM v0.19.0 is the latest release as of April 2026

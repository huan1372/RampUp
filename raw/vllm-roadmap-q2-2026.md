# vLLM Roadmap Q2 2026
- Source: https://github.com/vllm-project/vllm/issues/39749
- Ingested: 2026-04-14
- Type: GitHub issue / roadmap

Key points extracted:
- Custom Helion kernels planned for default usage
- torch.compile + CUDA streams support (PyTorch 2.12)
- Unwrapping custom ops (MLA, Fused MoE) for Inductor optimization
- KV cache manager rethink for complex layouts
- CPU offloading + Disk + connector API
- vLLM-Omni: disaggregated stages with different replica counts
- EPLB (Expert-Parallel Load Balancing) with async support
- GB200 optimization: NVLink, CPU unified memory, FP4, multi-stream

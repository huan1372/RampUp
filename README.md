# Inference Optimization Knowledge Base

A personal knowledge base for LLM inference optimization, following
[Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
Designed to be maintained by an AI agent (OpenClaw or Claude Code) with
[Obsidian](https://obsidian.md) as the human-readable frontend.

## Quick Start

### 1. Set up Obsidian
```bash
# Install Obsidian from https://obsidian.md
# Open this folder as an Obsidian vault:
#   File → Open Vault → select this directory (inference-kb/)
```

Enable these Obsidian community plugins for the best experience:
- **Dataview** — query pages by tags, dates, sources
- **Graph View** (built-in) — visualize cross-links between concepts
- **Obsidian Web Clipper** — save articles directly to `raw/`

### 2. Set up OpenClaw
```bash
# Install OpenClaw (requires Node.js)
git clone https://github.com/openclaw/openclaw.git
cd openclaw
npm install
openclaw onboard    # interactive setup wizard

# Copy the agent config
cp inference-kb/.openclaw/SOUL.md ~/.openclaw/SOUL.md

# Install the skill
openclaw skills add ./inference-kb/skills/inference-researcher/

# Set up cron jobs
openclaw cron add "0 8 * * *" "Run inference-researcher collect"
openclaw cron add "0 10 * * 0" "Run inference-researcher digest"
openclaw cron add "0 9 * * 5" "Run inference-researcher lint"
```

### 3. Alternative: Use with Claude Code
```bash
# If you prefer Claude Code over OpenClaw:
npm install -g @anthropic-ai/claude-code
cd inference-kb/

# Claude Code reads CLAUDE.md automatically as the project schema.
# You can manually trigger operations:
#   "ingest the new source I added to raw/"
#   "what are the latest vLLM optimizations for TTFT?"
#   "run a lint check on the wiki"
#   "write this week's digest"
```

## Structure
```
inference-kb/
├── CLAUDE.md                           ← schema (how the wiki works)
├── README.md                           ← this file
├── raw/                                ← immutable source documents
│   ├── vllm-roadmap-q2-2026.md
│   ├── vllm-releases.md
│   ├── vllm-benchmarks-2026.md
│   └── karpathy-llm-wiki.md
├── wiki/                               ← LLM-maintained knowledge base
│   ├── index.md                        ← master index
│   ├── overview.md                     ← high-level synthesis
│   ├── concepts/                       ← concept articles
│   │   ├── paged-attention.md
│   │   ├── continuous-batching.md
│   │   ├── chunked-prefill.md
│   │   ├── kv-cache-management.md
│   │   └── model-runner-v2.md
│   ├── techniques/                     ← optimization techniques
│   │   ├── fp8-quantization.md
│   │   ├── speculative-decoding.md
│   │   ├── prefix-caching.md
│   │   ├── disaggregated-serving.md
│   │   └── tensor-parallelism.md
│   ├── changelog/                      ← weekly digests
│   │   └── week-2026-04-14.md
│   └── _templates/                     ← page templates
│       ├── concept.md
│       ├── technique.md
│       └── weekly-digest.md
├── skills/
│   └── inference-researcher/           ← OpenClaw skill
│       ├── SKILL.md
│       ├── collect.sh
│       ├── digest.sh
│       └── lint.sh
└── .openclaw/
    └── SOUL.md                         ← agent personality config
```

## How to Add Sources

### Manually
Drop a markdown file into `raw/` with this frontmatter:
```yaml
# Title of the Source
- Source: <URL>
- Ingested: YYYY-MM-DD
- Type: paper | blog | commit | discussion

Key points extracted:
- ...
```

Then tell your agent: "Ingest the new source in raw/"

### Via Obsidian Web Clipper
1. Install the Obsidian Web Clipper browser extension
2. Configure it to save to `raw/` in your vault
3. Clip interesting articles — the agent will ingest them on the next daily run

### Automatically (with OpenClaw cron)
The daily cron job searches for new content and saves it to `raw/` automatically.

## Querying the Knowledge Base
Ask your agent questions like:
- "What are the latest vLLM optimizations for reducing TTFT?"
- "Compare chunked prefill strategies across the last month's papers"
- "What changed in vLLM's scheduler since v0.6?"
- "How does FP8 quantization affect KV cache quality?"
- "What's the competitive landscape: vLLM vs SGLang vs TensorRT-LLM?"

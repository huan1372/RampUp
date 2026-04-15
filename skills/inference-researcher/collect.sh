#!/bin/bash
# Daily Inference Research Collection Script
# Run by OpenClaw cron: 0 8 * * *
#
# This script is a template — OpenClaw will execute it via its skill runner.
# The actual web searches and API calls happen through OpenClaw's tool system
# (web search, GitHub API, etc.), not via curl/wget.
#
# What this script does:
# 1. Tells the agent what to search for
# 2. The agent uses its tools to find new content
# 3. New sources are saved to raw/
# 4. Wiki is updated following CLAUDE.md schema

set -euo pipefail

WIKI_ROOT="${WIKI_ROOT:-$(dirname "$0")/../../}"
DATE=$(date +%Y-%m-%d)

echo "=== Inference Research Collection — $DATE ==="
echo ""
echo "AGENT INSTRUCTIONS:"
echo ""
echo "1. Search for new vLLM content from the past 24 hours:"
echo "   - GitHub vllm-project/vllm: new commits, issues, PRs tagged performance/optimization"
echo "   - blog.vllm.ai: new posts"
echo "   - arXiv: papers on inference optimization, KV cache, speculative decoding"
echo "   - Reddit r/LocalLLaMA and HN: vLLM or inference optimization discussions"
echo ""
echo "2. For each relevant finding:"
echo "   - Save to raw/${DATE}-<slug>.md with source URL and type in frontmatter"
echo "   - Run the INGEST workflow from CLAUDE.md"
echo ""
echo "3. After all sources are ingested:"
echo "   - Run the LINT workflow"
echo "   - Report what was added/updated"
echo ""
echo "Wiki root: $WIKI_ROOT"

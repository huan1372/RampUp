# Inference Researcher Skill

An OpenClaw skill that automatically discovers, collects, and compiles inference optimization knowledge into a structured LLM wiki, following Karpathy's LLM Knowledge Base pattern.

## Description
This skill searches for the latest vLLM commits, blog posts, papers, and community discussions about LLM inference optimization. It ingests raw sources and compiles them into a structured markdown wiki with cross-links and index updates.

## Triggers
- Cron: daily at 08:00 UTC (collection) and weekly on Sundays at 10:00 UTC (digest)
- Manual: "research inference", "update wiki", "what's new in vLLM"

## Workflow: Daily Collection

1. **Search** for new content (past 24 hours):
   - vLLM GitHub: new commits to `main`, new issues tagged `performance` or `optimization`
   - vLLM blog (blog.vllm.ai): new posts
   - arXiv: papers mentioning "vLLM", "inference optimization", "KV cache", "speculative decoding"
   - Hacker News / Reddit r/LocalLLaMA: discussions about inference performance

2. **Filter** for relevance:
   - Must relate to inference optimization, serving performance, or vLLM internals
   - Skip: marketing, unrelated model releases, non-technical discussions

3. **Save** raw sources to `raw/` with metadata frontmatter:
   ```
   raw/YYYY-MM-DD-<slug>.md
   ```

4. **Ingest** into wiki following the CLAUDE.md schema:
   - Read new source
   - Update or create concept/technique pages
   - Update index.md and overview.md
   - Add cross-links

## Workflow: Weekly Digest

1. Scan all wiki pages with `updated` date in the past 7 days
2. Compile a summary of what changed, what was learned, any contradictions
3. Write to `wiki/changelog/week-YYYY-MM-DD.md`
4. Post digest summary to the configured channel (Slack/Discord)

## Workflow: Lint

Run after every ingest cycle:
1. Check all cross-links resolve
2. Flag pages with no sources
3. Flag pages not updated in 30+ days
4. Report contradictions between pages
5. Log results to `wiki/changelog/`

## Configuration

```yaml
# In OpenClaw config
skill: inference-researcher
schedule:
  collect: "30 22 * * *"    # daily at 22:30 UTC (3:30 PM PDT) — temporary for testing
  digest: "0 23 * * 3"      # Wednesday at 23:00 UTC (4:00 PM PDT) — temporary for testing
  lint: "30 22 * * 3"       # Wednesday at 22:30 UTC (3:30 PM PDT) — temporary for testing

sources:
  github:
    - repo: vllm-project/vllm
      watch: [commits, issues, releases]
      labels: [performance, optimization, scheduler, memory]
  blogs:
    - url: https://blog.vllm.ai/
    - url: https://developers.redhat.com/topics/ai
  arxiv:
    queries:
      - "vLLM inference optimization"
      - "KV cache management LLM"
      - "speculative decoding"
      - "LLM serving throughput"
  community:
    - reddit: r/LocalLLaMA
    - hackernews: search "vLLM OR inference optimization"

wiki_path: /path/to/inference-kb/
```

## Files
```
skills/inference-researcher/
├── SKILL.md          ← this file
├── collect.sh        ← daily collection script
├── digest.sh         ← weekly digest script
└── lint.sh           ← wiki health check script
```

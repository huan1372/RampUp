#!/bin/bash
# Weekly Digest Script
# Run by OpenClaw cron: 0 10 * * 0 (Sunday 10:00 UTC)

set -euo pipefail

WIKI_ROOT="${WIKI_ROOT:-$(dirname "$0")/../../}"
WEEK_START=$(date -d "7 days ago" +%Y-%m-%d)
DATE=$(date +%Y-%m-%d)

echo "=== Weekly Digest — $DATE ==="
echo ""
echo "AGENT INSTRUCTIONS:"
echo ""
echo "1. Find all wiki pages updated since $WEEK_START:"
echo "   - Check 'updated' field in YAML frontmatter of all pages in wiki/"
echo ""
echo "2. For each updated page, summarize:"
echo "   - What new information was added"
echo "   - What changed from previous version"
echo "   - Any contradictions with other pages"
echo ""
echo "3. Write digest to wiki/changelog/week-${DATE}.md using the template"
echo ""
echo "4. Include:"
echo "   - Open questions across the wiki"
echo "   - Suggested next reads for the coming week"
echo "   - Any themes or trends emerging"
echo ""
echo "5. Post a summary to the configured notification channel"
echo ""
echo "Wiki root: $WIKI_ROOT"

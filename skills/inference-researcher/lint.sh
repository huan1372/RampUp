#!/bin/bash
# Wiki Lint / Health Check Script
# Run by OpenClaw cron: 0 9 * * 5 (Friday 09:00 UTC)

set -euo pipefail

WIKI_ROOT="${WIKI_ROOT:-$(dirname "$0")/../../}"
DATE=$(date +%Y-%m-%d)

echo "=== Wiki Lint — $DATE ==="
echo ""
echo "AGENT INSTRUCTIONS:"
echo ""
echo "1. CROSS-LINK CHECK:"
echo "   - Scan all .md files in wiki/ for markdown links"
echo "   - Verify each link target exists"
echo "   - Report broken links"
echo ""
echo "2. SOURCE CHECK:"
echo "   - Verify every page in wiki/concepts/ and wiki/techniques/ has non-empty 'sources' frontmatter"
echo "   - Verify each referenced source exists in raw/"
echo "   - Flag pages with no sources"
echo ""
echo "3. STALENESS CHECK:"
echo "   - Flag pages with 'updated' date older than 30 days"
echo "   - Suggest which pages should be re-researched"
echo ""
echo "4. CONTRADICTION CHECK:"
echo "   - Compare claims across related pages"
echo "   - Flag any conflicting numbers, dates, or assertions"
echo ""
echo "5. INDEX CHECK:"
echo "   - Verify every page in wiki/concepts/ and wiki/techniques/ is listed in index.md"
echo "   - Flag any orphaned pages"
echo ""
echo "6. Report results and auto-fix what's possible (broken index entries, missing links)"
echo ""
echo "Wiki root: $WIKI_ROOT"

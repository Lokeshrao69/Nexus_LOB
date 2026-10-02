#!/usr/bin/env bash
# ==============================================================================
# Nexus-LOB: Automated Cloud Runner for 8 Remaining ITCH Dates
# ==============================================================================
set -euo pipefail

echo "==> 1. Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r python_quant/requirements.txt
pip install -r requirements-dev.txt

echo "==> 2. Processing 8 remaining dates (streaming slice + E1-E6 research)..."
# 15 completed dates are automatically skipped via --skip-existing.
# 7 retired 404 dates are omitted.
python python_quant/scripts/batch_research_itch.py \
    --days 12082025,12092025,12102025,12112025,12122025,05152026,05182026,06122026 \
    --symbols AAPL,QQQ \
    --skip-existing

echo "==> 3. Generating cross-day session manifest & aggregation..."
python python_quant/scripts/build_session_manifest.py

echo "==> 4. Running validation tests..."
python -m pytest python_quant/tests/ -q

echo "==> 5. Staging and committing research artifacts..."
git config user.name "ShrikarT"
git config user.email "132975062+ShrikarT@users.noreply.github.com"

# Stage ONLY research outputs, NEVER raw .itch tapes
git add docs/results/multi_day/ docs/results/multi_day_aggregation.json docs/results/multi_day_aggregation.md

# Safety check: ensure no binary tapes are staged
if git diff --cached --name-only | grep -E '\.(itch|gz)$'; then
    echo "ERROR: Attempted to stage raw ITCH binary data! Aborting commit."
    exit 1
fi

git commit -m "feat(research): add completed ITCH research results for 8 remaining dates" \
    --author="ShrikarT <132975062+ShrikarT@users.noreply.github.com>"

echo "==> 6. Pushing to branch feat/multi-day-research-15-days..."
git push origin feat/multi-day-research-15-days

echo "==> All 8 dates completed and pushed successfully!"

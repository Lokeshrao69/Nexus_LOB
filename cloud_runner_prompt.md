# Prompt for Cloud Agent

You are working on the Nexus_LOB repository: https://github.com/Lokeshrao69/Nexus_LOB
Branch: `feat/multi-day-research-15-days`

Task: Fetch, process, and validate the remaining 8 eligible NASDAQ ITCH 5.0 dates for AAPL and QQQ, run the E1–E6 microstructure research pipeline, update the multi-day manifests, and push the results.

Target 8 dates:
12082025, 12092025, 12102025, 12112025, 12122025, 05152026, 05182026, 06122026

(Note: 15 dates are already completed in docs/results/multi_day/, and 7 historical 2018 dates are 404-retired on NASDAQ EMI).

Steps:
1. Ensure Python dependencies are installed:
   pip install -r python_quant/requirements.txt -r requirements-dev.txt

2. Run the research pipeline for the 8 remaining dates:
   python python_quant/scripts/batch_research_itch.py --days 12082025,12092025,12102025,12112025,12122025,05152026,05182026,06122026 --symbols AAPL,QQQ --skip-existing

3. Rebuild cross-day aggregation and session manifest:
   python python_quant/scripts/build_session_manifest.py

4. Run regression tests:
   python -m pytest python_quant/tests/ -q

5. Stage ONLY the research outputs under docs/results/ (DO NOT stage data/itch/ or any raw .itch/.gz tapes):
   git add docs/results/multi_day/ docs/results/multi_day_aggregation.*
   git commit -m "feat(research): add completed ITCH research results for 8 remaining dates" --author="ShrikarT <132975062+ShrikarT@users.noreply.github.com>"
   git push origin feat/multi-day-research-15-days

Alternatively, you can simply run:
   bash run_remaining_itch.sh

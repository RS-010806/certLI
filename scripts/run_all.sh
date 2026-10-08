#!/usr/bin/env bash
# Full pipeline in the order used for the report (about 3 to 4 hours on 2 CPU cores).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
run() { echo "=== $(date -u +%H:%M:%S) $*"; "$@" 2>&1 | tee -a logs/run_all.log; }

for m in colbertv2 answerai-small; do
  run python src/encode.py $m scifact nfcorpus --bf16          # E0: emb/<model>/<corpus>/
  run python exp/run_exact.py $m scifact nfcorpus              # E1: Table 1 (also writes S_w1/S_idf score matrices)
  run python exp/run_structure.py $m scifact nfcorpus          # E2: Table 2
  run python exp/run_slack.py $m scifact nfcorpus              # E4: Table 4 (token level)
  for d in scifact nfcorpus; do
    run python exp/run_cert.py $m $d --suite main              # E3: Table 3; saves results/bounds/*.npz
  done
  run python exp/run_eigen.py $m scifact                       # E8: Table 6
done
run python exp/analyze_stop.py                                 # E6: Table 5, Figure 1
run python exp/analyze_oracle.py                               # E7: Table 5 (oracle)
run python exp/analyze_realized.py                             # E5: Table 4 (document level), Figure 2
run python exp/make_report_assets.py                           # tables/, figs/, results/final_numbers.json
(cd report && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null)
echo "done: report/report.pdf"

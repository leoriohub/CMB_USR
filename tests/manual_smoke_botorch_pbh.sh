#!/bin/bash
# Manual smoke test for optimize_pbh_botorch.py
#
# Runs the BoTorch optimizer with a tiny budget (n_init=4, n_trials=10)
# on the sub-solar target using the real MS solver pipeline.
# All output goes to /tmp so no project files are touched.
#
# Usage:
#   bash tests/manual_smoke_botorch_pbh.sh
#
# Requires:
#   - Conda env cmb-anomaly active
#   - git HEAD up to date with lab machine if running via ssh
#
# Exit codes:
#   0 — smoke test passed
#   1 — error during execution
#   2 — output validation failed

set -euo pipefail

LOG=/tmp/test_botorch_pbh_smoke.jsonl
OUTDIR=/tmp/test_botorch_pbh_plots
rm -f "$LOG"
rm -rf "$OUTDIR"

echo "========================================================================"
echo "  BoTorch PBH — Manual Smoke Test"
echo "========================================================================"
echo "Started: $(date)"
echo "Log:     $LOG"
echo "Plots:   $OUTDIR"
echo ""

# ---- Run optimizer with minimal budget --------------------------------------
echo ">>> Running optimizer (n_init=4, n_trials=10, q_batch=2) ..."
python scripts/optimize_pbh_botorch.py \
    --x-c-lo 0.78 --x-c-hi 0.79 \
    --c-lo 0.7 --c-hi 0.8 \
    --beta-lo 1e-5 --beta-hi 1e-4 \
    --chi0-lo 7.5 --chi0-hi 8.5 \
    --N-star-lo 60 --N-star-hi 70 \
    --zeta-c-lo 0.07 --zeta-c-hi 0.08 \
    --target subsolar \
    --n-init 4 --n-trials 10 --q-batch 2 \
    --output-dir "$OUTDIR" --log "$LOG" \
    --workers 4

OPT_EXIT=$?
echo ""
echo ">>> Optimizer exit code: $OPT_EXIT"
echo ""

if [ "$OPT_EXIT" -ne 0 ]; then
    echo "[FAIL] Optimizer exited with code $OPT_EXIT"
    exit 1
fi

# ---- Validate output --------------------------------------------------------
echo "========================================================================"
echo "  Output Validation"
echo "========================================================================"

N_LINES=0
if [ -f "$LOG" ]; then
    N_LINES=$(wc -l < "$LOG")
fi
echo "  Log lines:      $N_LINES"

N_PLOTS=0
if [ -d "$OUTDIR" ]; then
    N_PLOTS=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
fi
echo "  Plots generated: $N_PLOTS"

# Check that log has content
if [ "$N_LINES" -lt 1 ]; then
    echo "[FAIL] Log file is empty or missing"
    exit 2
fi

# Check that log lines are valid JSON
INVALID=0
while IFS= read -r line; do
    if ! echo "$line" | python -m json.tool > /dev/null 2>&1; then
        INVALID=$((INVALID + 1))
    fi
done < "$LOG"

if [ "$INVALID" -gt 0 ]; then
    echo "[FAIL] $INVALID log lines are not valid JSON"
    exit 2
fi

# Check for at least one success entry
N_SUCCESS=$(python -c "
import json
n = 0
with open('$LOG') as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('status') == 'success':
            n += 1
print(n)
")
echo "  Success entries: $N_SUCCESS"

if [ "$N_SUCCESS" -lt 1 ]; then
    echo "[WARN] No success entries — all configs may have failed pre-filter"
fi

# Show summary
echo ""
echo "========================================================================"
echo "  Summary"
echo "========================================================================"
echo "  Log:         $LOG ($N_LINES lines, $N_SUCCESS success)"
echo "  Plots:       $OUTDIR ($N_PLOTS PNG files)"
echo "========================================================================"
echo "  Smoke test PASSED"
echo "========================================================================"

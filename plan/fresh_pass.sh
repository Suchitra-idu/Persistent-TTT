#!/usr/bin/env bash
# The preregistered fresh pass (plan/PREREGISTRATION.md), run from a clean worktree of the
# `fresh-pass` tag so every manifest records that commit. Sequential: the first model builds
# and saves the cases, the rest reload them. Run only after `fresh-pass` is pushed.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODAL="$REPO/.venv/bin/modal"
WT="${FRESH_WORKTREE:-$REPO/../Persistent-TTT-fresh-pass}"
OUT="$REPO/results"

git -C "$REPO" fetch --tags origin >/dev/null 2>&1 || true
git -C "$REPO" fetch origin main >/dev/null
if ! git -C "$REPO" merge-base --is-ancestor 'fresh-pass^{commit}' origin/main; then
  echo "the fresh-pass commit is not on origin/main; push it first" >&2
  exit 1
fi
[ -d "$WT" ] || git -C "$REPO" worktree add --detach "$WT" fresh-pass
[ "$(git -C "$WT" rev-parse HEAD)" = "$(git -C "$REPO" rev-parse 'fresh-pass^{commit}')" ]
[ -z "$(git -C "$WT" status --porcelain)" ]

CARRY="update_rule=hebbian,state_aware_gate=false,limit_docs=4000,carry_gate=true,carry_backprop=true,output_gate_bias_init=-6,eval_holdout_seed=1"
WIKI=(--n-wiki-shards 2 --wiki-first-shard 2 --wiki-case-seed 2027 --n-related 200 --unique-related
      --n-generic 200 --in-context --chain-lengths 1,2,5,10 --overlap-seeds 42,43
      --cases-file fresh)
COMMON=(--exclude-holdout-seed 0 --n-boot 10000 --tag fresh --out-dir "$OUT")

run() {
  echo "$(date -Is) start $1"
  # The editable install maps `ttt` to $REPO; PYTHONPATH makes the image ship the tag's code.
  (cd "$WT" && PYTHONPATH="$WT" "$MODAL" run --detach ttt/experiments/paper_eval_v1.py --checkpoint "$1/step_600" "${@:2}")
  echo "$(date -Is) done $1"
}

run hebbian-carry-600 --suites relatedness,regimes "${WIKI[@]}" "${COMMON[@]}" \
  --flags "$CARRY,run_name=hebbian-carry-600"
run hebbian-carry-600-s43 --suites relatedness,regimes "${WIKI[@]}" "${COMMON[@]}" \
  --flags "$CARRY,seed=43,run_name=hebbian-carry-600-s43"
run mixed-carry-600 --suites relatedness,regimes "${WIKI[@]}" "${COMMON[@]}" \
  --flags "$CARRY,strategy=mixed,run_name=mixed-carry-600"
run lora-only-matched --suites regimes "${COMMON[@]}" \
  --flags "num_layers=1,limit_docs=4000,eval_holdout_seed=1,run_name=lora-only-matched"

latest() { ls -d "$OUT/$1/step_600/"*-fresh | sort | tail -1; }
(cd "$REPO" && .venv/bin/python -m ttt.experiments.mixed_probe_v1 \
  "$(latest hebbian-carry-600)" "$(latest mixed-carry-600)")
(cd "$REPO" && .venv/bin/python -m ttt.experiments.compare_checkpoints_v1 --n-boot 10000 \
  "$(latest lora-only-matched)" "$(latest hebbian-carry-600)")
(cd "$REPO" && .venv/bin/python -m ttt.experiments.compare_checkpoints_v1 --n-boot 10000 \
  "$(latest lora-only-matched)" "$(latest hebbian-carry-600-s43)")

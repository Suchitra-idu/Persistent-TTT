#!/usr/bin/env bash
# Usage: pipeline.sh <profile> <run_name> <stop_step|0> <train_flags> [eval_args_file]
# Trains <run_name> on <profile>, stops the app once step_<stop_step> is saved (0 = run to the
# end of the schedule), then runs each line of eval_args_file as a paper_eval_v1 invocation.
set -u
PROFILE="$1"; RUN="$2"; STOP="$3"; FLAGS="$4"; EVALS="${5:-}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MODAL="$REPO/.venv/bin/modal"
LOGS="$REPO/plan/tier1/logs"; mkdir -p "$LOGS"
LOG="$LOGS/$RUN.log"
export MODAL_PROFILE="$PROFILE"
cd "$REPO"

note() { echo "$(date -Is) $*" | tee -a "$LOG"; }

if "$MODAL" volume ls ttt-checkpoints "$RUN" >/dev/null 2>&1; then
  note "refusing: $RUN already exists on $PROFILE"; exit 1
fi

note "train start on $PROFILE: $FLAGS"
"$MODAL" run --detach ttt/experiments/train_v1.py --flags "$FLAGS" >> "$LOGS/$RUN.train.log" 2>&1 &
TRAIN_PID=$!
APP=""
for _ in $(seq 60); do
  APP=$(grep -oE "ap-[A-Za-z0-9]{10,}" "$LOGS/$RUN.train.log" | head -1)
  [ -n "$APP" ] && break; sleep 10
done
[ -z "$APP" ] && { note "no app id found"; exit 1; }
echo "$APP" > "$LOGS/$RUN.app"
note "train app $APP"

if [ "$STOP" != "0" ]; then
  until "$MODAL" volume ls ttt-checkpoints "$RUN/step_$STOP" 2>/dev/null | grep -q ttt_params.pt; do
    kill -0 "$TRAIN_PID" 2>/dev/null || { note "train exited before step_$STOP"; exit 1; }
    sleep 90
  done
  note "step_$STOP saved; stopping $APP"
  "$MODAL" app stop -y "$APP" >> "$LOG" 2>&1
fi
wait "$TRAIN_PID"
note "train finished (exit $?)"

[ -z "$EVALS" ] && exit 0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  note "eval start: $line"
  eval "\"$MODAL\" run --detach ttt/experiments/paper_eval_v1.py $line" >> "$LOGS/$RUN.eval.log" 2>&1
  note "eval done (exit $?)"
done < "$EVALS"
note "pipeline complete"

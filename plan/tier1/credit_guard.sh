#!/usr/bin/env bash
# Usage: credit_guard.sh <profile> <limit_usd>
# Every 5 minutes, reads this month's compute cost; at or above <limit_usd> it stops every
# running app on <profile>, so the workspace never reaches its credit cap mid-run.
set -u
PROFILE="$1"; LIMIT="$2"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MODAL="$REPO/.venv/bin/modal"
LOG="$REPO/plan/tier1/logs/guard_$PROFILE.log"
export MODAL_PROFILE="$PROFILE"
while true; do
  cost=$("$MODAL" billing summary --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
def find(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if 'ephemeral' in k.lower() and isinstance(v,(int,float,str)):
                try: return float(v)
                except ValueError: pass
            r=find(v)
            if r is not None: return r
    if isinstance(o,list):
        for v in o:
            r=find(v)
            if r is not None: return r
print(find(d) or 0)")
  running=$("$MODAL" app list --json 2>/dev/null | python3 -c "
import json,sys;print(' '.join(a['app_id'] for a in json.load(sys.stdin) if a['state']!='deployed' and not a['stopped_at']))")
  echo "$(date -Is) cost=$cost running=[$running]" >> "$LOG"
  if python3 -c "import sys;sys.exit(0 if float('$cost')>=float('$LIMIT') else 1)"; then
    for app in $running; do "$MODAL" app stop -y "$app" >> "$LOG" 2>&1; done
    echo "$(date -Is) LIMIT $LIMIT reached; stopped [$running]" >> "$LOG"
    exit 0
  fi
  sleep 300
done

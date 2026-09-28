#!/usr/bin/env bash
# 长跑守护：这个环境会周期性回收长跑进程（主仓库 OWNERSHIP.md 实测记录），
# 从工具调用里起的后台进程会被回收，必须挂守护循环、且完成即退出。
#
# 用法: scripts/supervise.sh <名字> <命令...>
#   - 命令正常结束（退出码 0）→ 守护退出，不再重启
#   - 命令失败 → 退避后重试，最多 N 次
#   - 心跳与状态落 runs/logs/<名字>.status
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="$1"; shift
mkdir -p "$ROOT/runs/logs"
log="$ROOT/runs/logs/$name.log"
status="$ROOT/runs/logs/$name.status"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PATH="/home/wuwenjie01/majiang_ai/.venv/bin:$PATH"

max_tries="${SUPERVISE_MAX_TRIES:-20}"
try=0
while :; do
  try=$((try + 1))
  echo "$(date -Is) start try=$try" >>"$log"
  set +e
  nice -n 15 python -u "$@" >>"$log" 2>&1
  code=$?
  set -e
  echo "$(date -Is) exit=$code try=$try" >>"$log"
  if [ "$code" -eq 0 ]; then
    echo "$(date -Is) DONE try=$try" >"$status"
    exit 0
  fi
  if [ "$try" -ge "$max_tries" ]; then
    echo "$(date -Is) GIVEUP try=$try code=$code" >"$status"
    exit "$code"
  fi
  echo "$(date -Is) retry_in_10s try=$try code=$code" >"$status"
  sleep 10
done

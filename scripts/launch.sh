#!/usr/bin/env bash
# 启动一个后台实验（nohup + 落日志），避免依赖 exec 会话生命周期。
# 用法: scripts/launch.sh <名字> <命令...>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="$1"; shift
mkdir -p "$ROOT/runs/logs"
log="$ROOT/runs/logs/$name.log"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PATH="/home/wuwenjie01/majiang_ai/.venv/bin:$PATH"
nohup nice -n 15 python -u "$@" >"$log" 2>&1 &
echo "pid=$! name=$name log=$log"

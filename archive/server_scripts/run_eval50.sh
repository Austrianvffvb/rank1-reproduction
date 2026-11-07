#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/rank1-finetune

/root/autodl-tmp/rank1-env/bin/python -u scripts/eval_50.py base
/root/autodl-tmp/rank1-env/bin/python -u scripts/eval_50.py finetuned

echo "两个模型的评估均已完成"

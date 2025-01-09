# scripts/run_all.sh

#!/bin/bash

# エラーハンドリング
set -e

# 評価の実行
echo "Running mt-bench evaluation..."
python scripts/run_jmtbench_eval.py

echo "Evaluation completed successfully."

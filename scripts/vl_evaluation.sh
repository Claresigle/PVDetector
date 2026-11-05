#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -B Evaluate_experiment_VL.py \
  --experiment_name vl_evaluation \
  --output_dir ./result/evaluate_result/ \
  --log_dir ./logs/evaluate_result/ \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset_name Alpaca \
  --role_name RecipeMaster \
  --pv_concept_source_role RecipeMaster \
  --pv_concept_source_model Qwen2.5-VL-7B-Instruct \
  --pv_source_sample_num 30 \
  --device cuda:1 \
  --dtype float16 \
  --seed 38 \
  --fpr_target 0.01

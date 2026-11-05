#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -B Evaluate_experiment.py \
  --experiment_name text_evaluation \
  --output_dir ./result/evaluate_result/ \
  --log_dir ./logs/evaluate_result/ \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name Alpaca \
  --role_name RecipeMaster \
  --pv_concept_source_role RecipeMaster \
  --pv_concept_source_model Meta-Llama-3.1-8B-Instruct \
  --pv_source_sample_num 30 \
  --device cuda:1 \
  --dtype float16 \
  --seed 38 \
  --fpr_target 0.01

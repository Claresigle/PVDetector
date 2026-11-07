# PVDetector

Official implementation of **PVDetector: Detecting Prompt Injection Attacks on Purpose-Specific LLM Agents through Policy-Violation Concept Analysis** (ACM Multimedia 2026).

[Paper](https://doi.org/10.1145/3767308.3836398) | [arXiv](https://arxiv.org/abs/2607.12624) | [License](LICENSE)

PVDetector is a training-free detector for prompt injection attacks against purpose-specific LLM agents. It extracts policy-violation concepts from a small set of compliant and violating queries, then detects suspicious inputs by measuring their alignment with these concepts in the model's hidden states.

![PVDetector overview](README.assets/PVDetector_overview.png)

Model adapters are provided for Llama-3.1-8B, Qwen2.5-7B/14B, Qwen2.5-VL-7B, and Phi-3.5 Vision. Agent scenarios include RecipeMaster, TripPlanner, CareerCounselor, and LineMonitor.

## Installation

Use Python 3.10 and run all commands from the repository root. Install the two requirements files in separate environments:

| Environment | Tasks | Requirements | Transformers |
|---|---|---|---|
| `pvdetector` | PV extraction, layer localization, evaluation, VLM attacks | [requirements.txt](requirements.txt) | 4.57.3 |
| `pvdetector-attack` | Text attacks | [Attack/requirements.txt](Attack/requirements.txt) | 4.47.1 |

**Evaluation and VLM attacks:**

```bash
conda create -n pvdetector python=3.10 -y
conda activate pvdetector
pip install -r requirements.txt
```

**Text attacks (nanoGCG):**

```bash
conda create -n pvdetector-attack python=3.10 -y
conda activate pvdetector-attack
pip install -r Attack/requirements.txt
```

The text attack environment uses Transformers 4.47.1 to satisfy [nanoGCG's dependency constraint](https://github.com/GraySwanAI/nanoGCG/blob/main/pyproject.toml). VLM attacks use the project's adapted backends with Transformers 4.57.3. nanoGCG is bundled in `Attack/`; no separate `pip install nanogcg` is needed.

## Data Preparation

### 1. System prompts and PV concept samples

- **System prompts:** `datasets/Agents_system_prompt/` contains the role definitions and restriction policies for each agent.
- **Contrastive sample pairs:** each pair contains a benign query and a policy-violating (PV) query for extracting PV concepts. Samples are stored in `pv_detector/PV_concept_extract/samples/<role>/{benign_samples,pv_samples}/Valid_for_<model>/30_shot/`.

The sample files are named `benign_samples.json` and `pv_samples.json`. Text entries are strings; VLM entries contain `msg` and `img_path`, with image paths relative to the repository root.

### 2. Evaluation data

- **Normal data:** benign queries that comply with the agent's policies, used for threshold calibration and false-positive rate evaluation.
- **Attack data:** prompt injection inputs constructed from policy-violating queries, used for false-negative rate evaluation.

Configure data paths in `configs/dataset_configs/<role>_datasets_config.json` or `<role>_VL_datasets_config.json`. Text evaluation files contain prompt strings. VLM records contain `msg` (or `Question`) and `img_path`, preserving the original image mapping after filtering or sampling. VLM string lists require `img_dir` with images named `img_1.jpg`, `img_2.jpg`, etc., in output order.

Prepare Normal data and rejected source queries before generating attacks. The attack runners read `Questions_rejected_*it.json` from the corresponding `Rejected_by_<model>` dataset directory. Generate attack data through the unified entrypoint:

```bash
python -m Attack.run_attack {gcg,iggcg,manual} {text,vl} [options]
python -m Attack.run_attack gcg text --help
```

The `manual` runner implements the heuristic-based attacks described in the paper. Select `Ignore`, `Fake_Complete`, or `Combine` through `--attack_strategy`.

Generate the sample counts named in the evaluation config. Text Manual attacks on Alpaca / MMLU require `--sample_num 300`:

| Modality / dataset | Manual | GCG / IgGCG |
|---|---|---|
| Text: Alpaca / MMLU | 300 | 100 |
| Text: Leak_Prompt / Unauthorized_Action | 100 | 100 |
| VL: Alpaca | 80 (Qwen) / 81 (Phi) | 80 (Qwen) / 81 (Phi) |

Text GCG and IgGCG default to 100 samples. All VL attack methods default to all available rejected samples, currently 80 for Qwen and 81 for Phi. Examples matching the RecipeMaster evaluation configs:

```bash
conda activate pvdetector-attack
python -m Attack.run_attack manual text \
  --model meta-llama/Llama-3.1-8B-Instruct --role_name RecipeMaster \
  --dataset Alpaca --attack_strategy Combine --sample_num 300 --device cuda:0

python -m Attack.run_attack gcg text \
  --model meta-llama/Llama-3.1-8B-Instruct --role_name RecipeMaster \
  --dataset Alpaca --sample_num 100 --device cuda:0

conda activate pvdetector
python -m Attack.run_attack iggcg vl \
  --model Qwen/Qwen2.5-VL-7B-Instruct --role_name RecipeMaster \
  --dataset Alpaca --sample_num 80 --device cuda:0
```

Outputs default to `datasets/Attack_Data/<role>/<dataset>/<attack>_attack/<model>/`. IgGCG uses `Ignore_GCG_attack` as its storage directory name. Evaluation configs reference the all-results JSON files. Prepare these files and extract the corresponding PV vectors before evaluation.

## Usage

`--model` defaults to an official Hugging Face model ID. To use a local checkpoint, pass `--model /path/to/Llama-3.1-8B-Instruct` instead. Keep the local directory's basename consistent with the model name; both `Llama-3.1-8B-Instruct` and the legacy `Meta-Llama-3.1-8B-Instruct` map to the existing Llama configuration and vector directories.

### 1. Extract PV vectors

```bash
python -m pv_detector.extract_pv_concept \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --role_name RecipeMaster \
  --sample_num 30 \
  --device cuda:0
```

Mean-difference PV vectors are saved to `pv_detector/PV_concept_extract/pv_vectors/<model>/<role>/30_shot/layers_pv_vector_mean.pt` and used throughout extraction, layer localization, and evaluation.

For VLMs, use the same command with the corresponding model ID or local path and `--modality vl`.

### 2. Evaluate a text model

```bash
python Evaluate_experiment.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --role_name RecipeMaster \
  --dataset_name Alpaca \
  --pv_concept_source_model Meta-Llama-3.1-8B-Instruct \
  --estimate_threshold True \
  --fpr_target 0.01 \
  --device cuda:0
```

`--estimate_threshold` defaults to `True`: calibrate once from up to 100 benign samples using the current PV vectors and key layers. The default target FPR is 0.01 (1%); `--fpr_target` accepts a single value for text and VLM evaluation.  Recalibrate after changing vector contents. Add `--locate_key_layers True` to select text-model key layers dynamically with calibration enabled.

### 3. Evaluate a vision-language model

Extract vectors for the VLM first, then run:

```bash
python Evaluate_experiment_VL.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --role_name RecipeMaster \
  --dataset_name Alpaca \
  --pv_concept_source_model Qwen2.5-VL-7B-Instruct \
  --estimate_threshold True \
  --fpr_target 0.01 \
  --device cuda:0
```

When changing the agent role, also set `--pv_concept_source_role` to the role used for vector extraction. Evaluation CSVs contain false-positive and false-negative rates (FPR/FNR) and are saved under `result/evaluate_result/`; logs are saved under `logs/evaluate_result/`.

### Optional: inspect violation-aware layers

```bash
python -m pv_detector.locate_pv_aware_layers \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --role_name RecipeMaster \
  --device cuda:0
```

This command reports the selected layers and saves a visualization. Use `--help` on any entrypoint to inspect all available options.

## Citation

```bibtex
@inproceedings{wang2026pvdetector,
  author={Junhui Wang and Hangtao Zhang and Zhirun Zheng and Li Zeng and Jiejun Xiao and Xi Luo and Lihua Yin and Saiqin Long},
  title     = {PVDetector: Detecting Prompt Injection Attacks on Purpose-Specific LLM Agents through Policy-Violation Concept Analysis},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3836398}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party components retain their upstream licenses and citation requirements.

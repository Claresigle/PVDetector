# -*-coding:utf-8-*-
import glob
import os
import json
import random
from pathlib import Path
from typing import List, Any

from vl_dataset import load_vl_samples


def load_json_file(file_path: str) -> List[Any]:
    """Read JSON file（Used for list data）"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Invalid JSON format; expected a list: {file_path}")
            return data
        except json.JSONDecodeError:
            raise ValueError(f"JSON parsing failed: {file_path}")


def save_json_file(file_path: str, data: Any):
    # Write a complete temporary file, then replace atomically to avoid partial JSON.
    temp_path = f"{file_path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_system_prompts(domain_dir):
    """Load system prompt words"""
    domain_dir = Path(domain_dir)
    system_prompts = {  # The dictionary stores system prompts corresponding to different fields.
        md_file.stem: md_file.read_text(encoding="utf-8").strip()
        for md_file in domain_dir.glob("*.md")
    }
    return system_prompts


def load_samples_from_dataset(data_path, sample_num=None):
    # Load JSON format data
    msg_data = load_json_file(data_path)
    if sample_num is not None:
        if sample_num == -1:
            samples_msg = msg_data
        else:
            if len(msg_data) < sample_num:
                raise ValueError(f"Not enough samples: sample_num={sample_num}")
            samples_msg = random.sample(msg_data, sample_num)
    else:
        samples_msg = msg_data
    # GCG optimizes target-response loss; the target is not part of the attack input.
    attack_target = "Sure, I can help with that."
    return samples_msg, attack_target


def load_samples_from_dataset_VL(data_path, sample_num=None):
    samples_data = load_vl_samples(data_path, sample_num=sample_num)
    attack_target = "Sure, I can help with that."
    return samples_data, attack_target


DATASET_CONFIGS = {
    "Alpaca": {
        "data_subdir": "Out_of_domain_data/Alpaca",
        "output_subdir": "Alpaca",
        "output_prefix": "alpaca",
    },
    "MMLU": {
        "data_subdir": "Out_of_domain_data/MMLU",
        "output_subdir": "MMLU",
        "output_prefix": "mmlu",
    },
    "Leak_Prompt": {
        "data_subdir": "Policy_violating_queries_data",
        "output_subdir": "Leak_Prompt",
        "output_prefix": "pv",
    },
    "Unauthorized_Action": {
        "data_subdir": "Policy_violating_queries_data",
        "output_subdir": "Unauthorized_Action",
        "output_prefix": "pv",
    },
}


def build_dataset_paths(args, role, model_name, attack_type="GCG"):
    # Attack rejected queries and organize outputs by role, dataset, attack, and model.
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"{args.dataset} is not supported.")

    config = DATASET_CONFIGS[args.dataset]
    data_path = os.path.join(
        args.tar_data_dir,
        role,
        config["data_subdir"],
        f"Rejected_by_{model_name}",
        "Questions_rejected_*it.json",
    )
    data_path = (glob.glob(data_path) or [None])[0]
    save_dir = os.path.join(
        args.save_dir,
        role,
        config["output_subdir"],
        f"{attack_type}_attack",
        model_name,
    )
    output_prefix = config["output_prefix"]
    save_path_no_deny = os.path.join(
        save_dir,
        f"{output_prefix}_{args.sample_num}it_attack_data_no_deny.json",
    )
    save_path = os.path.join(
        save_dir,
        f"{output_prefix}_{args.sample_num}it_attack_data.json",
    )
    return data_path, save_dir, save_path_no_deny, save_path


def build_dataset_paths_VL(args, role, model_name, attack_type="GCG"):
    config = DATASET_CONFIGS[args.dataset]
    data_path = os.path.join(
        args.tar_data_dir,
        f"{role}_data_for_VLM",
        config["data_subdir"],
        f"Rejected_by_{model_name}",
        "Questions_rejected_*it.json",
    )
    data_path = (glob.glob(data_path) or [None])[0]
    save_dir = os.path.join(
        args.save_dir,
        role,
        config["output_subdir"],
        f"{attack_type}_attack",
        model_name,
    )
    output_prefix = config["output_prefix"]
    save_path_no_deny = os.path.join(
        save_dir,
        f"{output_prefix}_{args.sample_num}it_attack_data_no_deny.json",
    )
    save_path = os.path.join(
        save_dir,
        f"{output_prefix}_{args.sample_num}it_attack_data.json",
    )
    return data_path, save_dir, save_path_no_deny, save_path


def resolve_vl_sample_num(args, role, model_name):
    """Resolve an omitted VL count before constructing log and output paths."""
    if args.sample_num is not None and args.sample_num != -1:
        return args.sample_num
    data_path, _, _, _ = build_dataset_paths_VL(args, role, model_name)
    if data_path is None:
        raise FileNotFoundError(
            f"No rejected VL dataset found for role={role}, model={model_name}."
        )
    sample_num = len(load_json_file(data_path))
    if sample_num == 0:
        raise ValueError(f"Rejected VL samples must not be empty: {data_path}")
    return sample_num

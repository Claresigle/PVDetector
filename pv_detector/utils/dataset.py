import csv
import json
import logging
import os
import random
from pathlib import Path

import numpy as np

from common.vl_dataset import load_vl_samples


def load_json_file(file_path: str):
    """Helper function."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Invalid JSON format; expected a list: {file_path}")
            return data
        except json.JSONDecodeError:
            raise ValueError(f"JSON parsing failed: {file_path}")


def get_normal_config(datasets_config):
    
    normal_config = {}
    for key, config in datasets_config.items():
        if "normal" in key.lower() or "ID" in key:
            normal_config[key] = config
    return normal_config


def save_results_csv(results, output_path):
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset Name", "FPR", "FNR"])
        fnrs = []
        for dataset_name, result in results.items():
            if result is not None:
                writer.writerow([dataset_name, f"{result[0]:.4f}", f"{result[1]:.4f}"])
            if result is not None and result[1] >= 0.0:
                fnrs.append(result[1])
        if fnrs:
            fnr = np.mean(fnrs)
            writer.writerow(["mean FNR", f"{fnr:.4f}"])
        else:
            writer.writerow(["mean FNR", "N/A"])
    logging.info(f"Results successfully written to {output_path}")


def load_system_prompts(domain_dir):
    """Load system prompts."""
    domain_dir = Path(domain_dir)
    system_prompts = {  
        md_file.stem: md_file.read_text(encoding="utf-8").strip()
        for md_file in domain_dir.glob("*.md")
    }
    return system_prompts


def load_config(config_file_path):
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"File not found: {config_file_path}")
    with open(config_file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            raise ValueError(f"JSON parsing failed: {config_file_path}")


def combine_normal_data(datasets):
    
    # Merge benign subsets for shared threshold calibration and FPR evaluation.
    normal_msg_list = []
    other_datasets = {}
    for key, dataset in datasets.items():
        if "normal" in key.lower() or "ID" in key:
            normal_msg_list.extend(dataset)
        else:
            other_datasets[key] = dataset
    
    other_datasets["Normal"] = normal_msg_list
    return other_datasets


def load_datasets_samples(datasets_config):
    """
    Args:
        datasets_config: dict():
        {
            "LLM-Normal": {"data_path": str, "select_num": int},
        },
    Returns:
        test_samples: dict:
        {
            "LLM-Normal": [
                {"msg": str, "label": int},
            ],
        },
    """
    # Use 0 for benign samples and 1 for attacks throughout evaluation.
    test_samples = {}
    for key, config in datasets_config.items():
        msg_list = load_samples_from_dataset(config["data_path"], config["sample_num"])
        if "normal" in key.lower() or "ID" in key:
            test_samples[key] = [{"msg": msg, "label": 0} for msg in msg_list]
        else:
            test_samples[key] = [{"msg": msg, "label": 1} for msg in msg_list]
    return test_samples


def load_VL_datasets_samples(datasets_config):
    """
    Args:
        datasets_config: dict():
        {
            "LLM-Normal": {"data_path": str, "sample_num": int},
        },
    Returns:
        test_samples: dict:
        {
            "LLM-Normal": [
                {"msg": str, "img_path": str, "label": int},
            ],
        },
    """
    test_samples = {}
    for key, config in datasets_config.items():
        data_list = load_samples_from_VL_dataset(
            config["data_path"], config.get("img_dir"), config["sample_num"]
        )
        if "normal" in key.lower() or "ID" in key:
            test_samples[key] = [{"msg": msg, "img_path": img_path, "label": 0} for msg, img_path in data_list]
        else:
            test_samples[key] = [{"msg": msg, "img_path": img_path, "label": 1} for msg, img_path in data_list]
    return test_samples


def load_samples_from_dataset(data_path, sample_num=None):
    
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
    return samples_msg


def load_samples_from_VL_dataset(data_path, img_dir=None, sample_num=None):
    return load_vl_samples(data_path, img_dir=img_dir, sample_num=sample_num)

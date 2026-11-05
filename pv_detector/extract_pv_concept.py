# -*-coding:utf-8-*-
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from model_config import DEFAULT_TEXT_MODEL_ID, MODEL_ARGUMENT_HELP, get_model_key
from pv_detector.utils.activations import compute_last_hidden_diff
from pv_detector.utils.dataset import load_json_file, load_system_prompts
from pv_detector.utils.model import (
    get_hidden_states,
    get_hidden_states_vl,
    is_vision_language_model,
    load_model,
)
from pv_detector.utils.other_utils import create_pairs, get_pv_vector_dir, set_logging, set_seed


PV_DETECTOR_DIR = Path(__file__).resolve().parent
REPO_DIR = PV_DETECTOR_DIR.parent
MODALITIES = ("auto", "text", "vl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text or vision-language PV vectors.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TEXT_MODEL_ID,
        help=MODEL_ARGUMENT_HELP,
    )
    parser.add_argument("--modality", choices=MODALITIES, default="auto")
    parser.add_argument("--domain_dir", type=str, default=str(REPO_DIR / "datasets" / "Agents_system_prompt"))
    parser.add_argument(
        "--sample_dir",
        type=str,
        default=str(PV_DETECTOR_DIR / "PV_concept_extract" / "samples"),
    )
    parser.add_argument("--role_name", type=str, default="LineMonitor")
    parser.add_argument("--sample_num", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--log_dir", type=str, default=str(PV_DETECTOR_DIR / "logs" / "extract_pv_concept"))
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(PV_DETECTOR_DIR / "PV_concept_extract" / "pv_vectors"),
    )
    return parser.parse_args()


def resolve_modality(model_name, requested_modality):
    inferred_modality = "vl" if is_vision_language_model(model_name) else "text"
    if requested_modality == "auto":
        return inferred_modality
    if requested_modality != inferred_modality:
        raise ValueError(
            f"Model {model_name!r} is recognized as {inferred_modality}, "
            f"but --modality={requested_modality} was requested"
        )
    return requested_modality


def validate_samples(samples, modality, source_name):
    if not samples:
        raise ValueError(f"{source_name} samples must not be empty")

    for index, sample in enumerate(samples):
        if modality == "text":
            if not isinstance(sample, str):
                raise TypeError(f"{source_name} sample {index} must be a string for text extraction")
            continue

        if not isinstance(sample, dict):
            raise TypeError(f"{source_name} sample {index} must be an object for VL extraction")
        missing_keys = {"msg", "img_path"} - sample.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"{source_name} sample {index} is missing required fields: {missing}")


def create_sample_pairs(role, model_name, sample_dir, sample_num=30, modality="text"):
    # Load both sample sets for the same model, role, and shot count before pairing.
    benign_samples_dir = Path(sample_dir) / role / "benign_samples" / f"Valid_for_{model_name}" / f"{sample_num}_shot"
    pv_samples_dir = Path(sample_dir) / role / "pv_samples" / f"Valid_for_{model_name}" / f"{sample_num}_shot"
    benign_samples = load_json_file(benign_samples_dir / "benign_samples.json")
    pv_samples = load_json_file(pv_samples_dir / "pv_samples.json")
    validate_samples(benign_samples, modality, "benign")
    validate_samples(pv_samples, modality, "PV")
    return create_pairs(benign_samples, pv_samples)


def get_sample_hidden_states(model, tokenizer_or_processor, sample, system_prompt, modality, model_name):
    if modality == "text":
        return get_hidden_states(model, tokenizer_or_processor, sample, system_prompt)
    return get_hidden_states_vl(
        model,
        tokenizer_or_processor,
        sample["msg"],
        sample["img_path"],
        system_prompt,
        model_name,
    )


def collect_layer_differences(model, tokenizer_or_processor, sample_pairs, system_prompt, modality, model_name):
    num_layers = model.config.num_hidden_layers
    layers_diff_matrix = [[] for _ in range(num_layers)]

    for benign_sample, pv_sample in tqdm(sample_pairs, total=len(sample_pairs), desc="Inference on sample pairs"):
        benign_hidden_states = get_sample_hidden_states(
            model,
            tokenizer_or_processor,
            benign_sample,
            system_prompt,
            modality,
            model_name,
        )
        pv_hidden_states = get_sample_hidden_states(
            model,
            tokenizer_or_processor,
            pv_sample,
            system_prompt,
            modality,
            model_name,
        )
        if len(benign_hidden_states) != num_layers or len(pv_hidden_states) != num_layers:
            raise ValueError(
                f"Expected {num_layers} hidden-state layers, got "
                f"benign={len(benign_hidden_states)} and PV={len(pv_hidden_states)}"
            )
        for layer_id in range(num_layers):
            # Each pair contributes a PV-minus-benign difference at the final token.
            layers_diff_matrix[layer_id].append(
                compute_last_hidden_diff(benign_hidden_states[layer_id], pv_hidden_states[layer_id])
            )

    return layers_diff_matrix


def compute_pv_vectors(layers_diff_matrix):
    """Compute each layer's mean PV-minus-benign hidden-state difference."""
    # Average differences per layer; the output shape is [num_layers, hidden_dim].
    return torch.stack([
        torch.mean(torch.stack(diff_matrix), dim=0)
        for diff_matrix in layers_diff_matrix
    ], dim=0)


def save_pv_vectors(save_dir, pv_vector_mean):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(pv_vector_mean, os.path.join(save_dir, "layers_pv_vector_mean.pt"))


def main():
    args = parse_args()
    model_name = get_model_key(args.model)
    modality = resolve_modality(model_name, args.modality)

    set_seed(args.seed)
    model, tokenizer_or_processor = load_model(args.model, args.device, args.dtype)
    system_prompts = load_system_prompts(args.domain_dir)
    if args.role_name not in system_prompts:
        raise ValueError(f"No system prompt found for role={args.role_name!r}")

    log_file = os.path.join(
        args.log_dir,
        args.role_name,
        model_name,
        modality,
        f"{args.sample_num}_shot",
    )
    set_logging(log_file, stdout=False, add_time=False)

    sample_pairs = create_sample_pairs(
        args.role_name,
        model_name,
        args.sample_dir,
        args.sample_num,
        modality=modality,
    )
    layers_diff_matrix = collect_layer_differences(
        model,
        tokenizer_or_processor,
        sample_pairs,
        system_prompts[args.role_name],
        modality,
        model_name,
    )
    pv_vector_mean = compute_pv_vectors(layers_diff_matrix)

    # Extraction, localization, and evaluation share this model/role/shot layout.
    save_dir = get_pv_vector_dir(
        args.save_dir,
        model_name,
        args.role_name,
        args.sample_num,
    )
    save_pv_vectors(save_dir, pv_vector_mean)
    print(f"PV vectors saved to {save_dir}")


if __name__ == "__main__":
    main()

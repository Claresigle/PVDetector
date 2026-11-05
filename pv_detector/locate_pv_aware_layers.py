# -*-coding:utf-8-*-
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nf
from tqdm import tqdm

from model_config import DEFAULT_TEXT_MODEL_ID, MODEL_ARGUMENT_HELP, get_model_key
from pv_detector.extract_pv_concept import (
    create_sample_pairs,
    get_sample_hidden_states,
    resolve_modality,
)
from pv_detector.utils.activations import compute_layer_pv_projection
from pv_detector.utils.dataset import load_system_prompts
from pv_detector.utils.model import load_model
from pv_detector.utils.other_utils import (
    get_pv_vector_dir,
    load_pv_hidden_vector,
    set_logging,
    set_seed,
)


PV_DETECTOR_DIR = Path(__file__).resolve().parent
REPO_DIR = PV_DETECTOR_DIR.parent

LOCATE_STRATEGIES = ("pos_neg_cos", "pv_concept")
MODALITIES = ("auto", "text", "vl")
PROJECTION_TYPES = ("proj", "cos")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Locate PV-aware layers from positive/negative activation pairs.")
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
    parser.add_argument("--activations_dir", type=str, default=str(PV_DETECTOR_DIR / "activations"))

    parser.add_argument("--role_name", type=str, default="LineMonitor")
    parser.add_argument("--locate_strategy", default="pv_concept", type=str, choices=LOCATE_STRATEGIES)

    parser.add_argument(
        "--pv_vector_dir",
        default=str(PV_DETECTOR_DIR / "PV_concept_extract" / "pv_vectors"),
        type=str,
    )
    parser.add_argument("--pv_source_sample_num", type=int, default=30)
    parser.add_argument("--proj_type", default="proj", type=str, choices=PROJECTION_TYPES)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--log_dir", type=str, default=str(PV_DETECTOR_DIR / "logs" / "locate_pv_aware_layer"))
    parser.add_argument(
        "--visual_dir",
        type=str,
        default=str(PV_DETECTOR_DIR / "result" / "visual_result" / "Safety_Layer_Locate"),
    )
    return parser


def parse_args():
    parser = build_arg_parser()
    return parser.parse_args()


def build_runtime_locate_args(source_args, modality):
    """Build locator arguments from evaluation arguments and locator defaults."""
    values = vars(build_arg_parser().parse_args([])).copy()
    for key in values:
        if hasattr(source_args, key):
            values[key] = getattr(source_args, key)
    values["modality"] = modality
    return argparse.Namespace(**values)


def build_visual_dir(args, model_name):
    return os.path.join(
        args.visual_dir,
        model_name,
        f"Custom_{args.role_name}",
        args.locate_strategy,
    )


def collect_pos_neg_activations(
        model,
        tokenizer_or_processor,
        sample_pairs,
        system_prompt,
        modality,
        model_name):
    pos_neg_activations = []
    for benign_sample, pv_sample in tqdm(
            sample_pairs,
            total=len(sample_pairs),
            desc=f"Inference on {modality} sample pairs"):
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
        if len(benign_hidden_states) != len(pv_hidden_states):
            raise ValueError(
                "benign and PV samples returned different hidden-state layer counts: "
                f"benign={len(benign_hidden_states)}, PV={len(pv_hidden_states)}"
            )

        # Cache only the final input token at each layer for reuse across strategies.
        benign_last_token = torch.stack([layer[:, -1, :] for layer in benign_hidden_states], dim=0)
        pv_last_token = torch.stack([layer[:, -1, :] for layer in pv_hidden_states], dim=0)
        pos_neg_activations.append((benign_last_token, pv_last_token))
    return pos_neg_activations


def load_or_compute_pos_neg_activations(
        args,
        model_name,
        model,
        tokenizer_or_processor,
        system_prompts,
        modality):
    # Separate caches by model, role, and shot count to avoid mixing sample sets.
    cache_path = os.path.join(
        args.activations_dir,
        "pos_neg_samples",
        model_name,
        f"Custom_{args.role_name}_{args.pv_source_sample_num}_shot.pt",
    )
    if os.path.exists(cache_path):
        logging.info(f"Loading cached pos-neg activations from {cache_path}")
        return torch.load(cache_path)

    sample_pairs = create_sample_pairs(
        args.role_name,
        model_name,
        args.sample_dir,
        args.pv_source_sample_num,
        modality=modality,
    )
    pos_neg_activations = collect_pos_neg_activations(
        model,
        tokenizer_or_processor,
        sample_pairs,
        system_prompts[args.role_name],
        modality,
        model_name,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(pos_neg_activations, cache_path)
    logging.info(f"Saved pos-neg activations to {cache_path}")
    return pos_neg_activations


def prepare_pv_concept_context(args, model_name):
    data_dir = get_pv_vector_dir(
        args.pv_vector_dir,
        model_name,
        args.role_name,
        args.pv_source_sample_num,
    )
    return {
        "pv_layers_hidden_vector": load_pv_hidden_vector(
            data_dir=data_dir,
        )
    }


def prepare_strategy_context(args, model_name, pv_layers_hidden_vector=None):
    if args.locate_strategy == "pv_concept":
        if pv_layers_hidden_vector is not None:
            return {"pv_layers_hidden_vector": pv_layers_hidden_vector}
        return prepare_pv_concept_context(args, model_name)
    return {}


def compute_layer_cos_similarity(benign_hidden_states, pv_hidden_states):
    layer_scores = []
    for layer_idx, benign_layer_hidden in enumerate(benign_hidden_states):
        score = nf.cosine_similarity(benign_layer_hidden.squeeze(), pv_hidden_states[layer_idx].squeeze(), dim=0)
        layer_scores.append(score.item())
    return layer_scores


def compute_pv_discrepancy(args, benign_hidden_states, pv_hidden_states, context):
    # Compare PV alignment between the two sample sets at each layer.
    benign_strengths = compute_layer_pv_projection(
        benign_hidden_states,
        context["pv_layers_hidden_vector"],
        projection_type=args.proj_type,
    )
    pv_strengths = compute_layer_pv_projection(
        pv_hidden_states,
        context["pv_layers_hidden_vector"],
        projection_type=args.proj_type,
    )
    return [pv_score - benign_score for benign_score, pv_score in zip(benign_strengths, pv_strengths)]


def compute_layer_scores(args, pos_neg_activations, context):
    samples_similarity = []
    samples_discrepancy = []

    for benign_hidden_states, pv_hidden_states in tqdm(
            pos_neg_activations,
            total=len(pos_neg_activations),
            desc="Process pos_neg_activations"):
        if args.locate_strategy == "pos_neg_cos":
            samples_similarity.append(compute_layer_cos_similarity(benign_hidden_states, pv_hidden_states))
        elif args.locate_strategy == "pv_concept":
            samples_discrepancy.append(compute_pv_discrepancy(args, benign_hidden_states, pv_hidden_states, context))
        else:
            raise ValueError(f"Invalid locate strategy: {args.locate_strategy}")

    return samples_similarity, samples_discrepancy


def format_layer_indices(scores, predicate):
    return ", ".join(str(i) for i, item in enumerate(scores) if predicate(item))


def extract_layer_indices(scores, predicate):
    return [i for i, item in enumerate(scores) if predicate(item)]


def visualize_layer_scores(scores, y_label, title, visual_dir):
    from pv_detector.utils.visual import visualize_array_as_bar

    visualize_array_as_bar(
        scores,
        y_label=y_label,
        title=title,
        save_dir=visual_dir,
    )


def report_pos_neg_similarity(samples_similarity, visual_dir):
    similarity_mean = np.stack(samples_similarity).mean(axis=0)
    visualize_layer_scores(
        similarity_mean,
        "Pos-Neg Similarity",
        "Pos-Neg Similarity Per Layer",
        visual_dir,
    )
    logging.info(f"Pos_Neg_similarity in each layer:\n{similarity_mean}\n")
    logging.info(
        "Safety awareness exists at layer: "
        + format_layer_indices(similarity_mean, lambda item: item < 0)
        + ".\n"
    )
    # The cosine strategy selects layers with negative mean pairwise similarity.
    return extract_layer_indices(similarity_mean, lambda item: item < 0)


def report_pv_discrepancy(args, samples_discrepancy, visual_dir):
    visual_dir = os.path.join(
        visual_dir,
        f"Custom_mean_hidden_vector_{args.proj_type}",
    )
    discrepancy_mean = np.stack(samples_discrepancy).mean(axis=0)
    visualize_layer_scores(
        discrepancy_mean,
        "PV Strength Discrepancy",
        "Pos-Neg PV Strength Discrepancy Per Layer",
        visual_dir,
    )
    logging.info(f"PV Discrepancy Vector:\n{discrepancy_mean}\n")
    logging.info(
        "PV awareness broadly exists at layer: "
        + format_layer_indices(discrepancy_mean, lambda item: item > 0)
        + ".\n"
    )
    logging.info(
        "The most safety-aware layers are: "
        + format_layer_indices(discrepancy_mean, lambda item: item >= discrepancy_mean[-1])
        + ".\n"
    )
    # Select layers whose mean discrepancy is at least that of the final layer.
    return extract_layer_indices(discrepancy_mean, lambda item: item >= discrepancy_mean[-1])


def report_results(args, samples_similarity, samples_discrepancy, visual_dir):
    np.set_printoptions(precision=4, suppress=True)
    if args.locate_strategy == "pos_neg_cos":
        return report_pos_neg_similarity(samples_similarity, visual_dir)
    elif args.locate_strategy == "pv_concept":
        return report_pv_discrepancy(args, samples_discrepancy, visual_dir)
    else:
        raise ValueError(f"Invalid locate strategy: {args.locate_strategy}")


def locate_pv_aware_key_layers(
        args,
        model_name=None,
        model=None,
        tokenizer=None,
        system_prompts=None,
        pv_layers_hidden_vector=None):
    if model_name is None:
        model_name = get_model_key(args.model)
    modality = resolve_modality(model_name, getattr(args, "modality", "auto"))
    # Reuse evaluation resources when supplied; load a model for standalone runs.
    if model is None or tokenizer is None:
        set_seed(args.seed)
        model, tokenizer = load_model(args.model, args.device, args.dtype)
    if system_prompts is None:
        system_prompts = load_system_prompts(args.domain_dir)
    if args.role_name not in system_prompts:
        raise ValueError(f"No system prompt found for role={args.role_name!r}")

    visual_dir = build_visual_dir(args, model_name)
    context = prepare_strategy_context(
        args,
        model_name,
        pv_layers_hidden_vector=pv_layers_hidden_vector,
    )
    pos_neg_activations = load_or_compute_pos_neg_activations(
        args,
        model_name,
        model,
        tokenizer,
        system_prompts,
        modality,
    )
    samples_similarity, samples_discrepancy = compute_layer_scores(args, pos_neg_activations, context)
    key_layers = report_results(args, samples_similarity, samples_discrepancy, visual_dir)
    logging.info(f"Located key layers: {key_layers}")
    return key_layers


def main():
    args = parse_args()
    set_seed(args.seed)
    set_logging(
        os.path.join(args.log_dir, get_model_key(args.model), args.role_name),
        stdout=True,
        add_time=False,
    )
    key_layers = locate_pv_aware_key_layers(args)
    print(f"Located key layers: {key_layers}")


if __name__ == "__main__":
    main()

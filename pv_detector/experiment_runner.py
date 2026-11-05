import logging
import os
from dataclasses import dataclass
from datetime import datetime

import torch

from model_config import get_model_key
from pv_detector.utils.dataset import (
    combine_normal_data,
    load_config,
    load_datasets_samples,
    get_normal_config,
    load_system_prompts,
    load_VL_datasets_samples,
    save_results_csv,
)
from pv_detector.utils.evaluate import evaluate_FNR, evaluate_FPR
from pv_detector.utils.model import load_model
from pv_detector.utils.other_utils import (
    get_pv_vector_dir,
    load_pv_hidden_vector,
    set_logging,
    set_seed,
)


# Layer indices are zero-based Transformer indices, excluding the embedding output.
TEXT_KEY_LAYERS = {
    "Meta-Llama-3.1-8B-Instruct": [31],
    "Qwen2.5-7B-Instruct": list(range(25, 28)),
    "Qwen2.5-14B-Instruct": list(range(36, 48)),
}

VL_KEY_LAYERS = {
    "Qwen2.5-VL-7B-Instruct": list(range(23, 28)),
    "Phi-3.5-vision-instruct": list(range(16, 32)),
}


@dataclass(frozen=True)
class ExperimentContext:
    model_name: str
    key_layers: list
    experiment_name: str
    log_dir: str
    output_dir: str
    timestamp: str


@dataclass
class ExperimentResources:
    model_name: str
    model: object
    tokenizer_or_processor: object
    system_prompts: dict
    pv_layers_hidden_vector: object


def get_key_layers(model_name, multimodal=False):
    layer_table = VL_KEY_LAYERS if multimodal else TEXT_KEY_LAYERS
    try:
        return list(layer_table[model_name])
    except KeyError as exc:
        raise ValueError(f"No saved key layers for model={model_name}") from exc


def format_key_layers(key_layers):
    if not key_layers:
        raise ValueError("key_layers must not be empty")
    if len(key_layers) == 1:
        return f"keylayers({key_layers[0]})"
    if key_layers == list(range(key_layers[0], key_layers[-1] + 1)):
        return f"keylayers({key_layers[0]}-{key_layers[-1]})"
    return "keylayers(" + ",".join(str(layer) for layer in key_layers) + ")"


def resolve_experiment_name(args, multimodal=False):
    experiment_name = getattr(args, "experiment_name", None)
    if experiment_name:
        return experiment_name
    return "vl_evaluation" if multimodal else "text_evaluation"


def build_context(args, key_layers, multimodal=False):
    model_name = get_model_key(args.model)
    key_layers = list(key_layers)
    layer_slug = format_key_layers(key_layers)
    experiment_name = resolve_experiment_name(args, multimodal=multimodal)

    common_parts = [experiment_name, model_name, args.role_name]
    log_dir = os.path.join(
        args.log_dir,
        *common_parts,
        f"mean_pv_hidden_proj_{layer_slug}",
    )
    output_dir = os.path.join(
        args.output_dir,
        *common_parts,
        f"pv_concept_source({args.pv_concept_source_role}_"
        f"{args.pv_concept_source_model})_{layer_slug}",
    )
    os.makedirs(output_dir, exist_ok=True)
    return ExperimentContext(
        model_name=model_name,
        key_layers=key_layers,
        experiment_name=experiment_name,
        log_dir=log_dir,
        output_dir=output_dir,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )


def build_pv_vector_dir(args):
    return get_pv_vector_dir(
        args.pv_vector_dir,
        args.pv_concept_source_model,
        args.pv_concept_source_role,
        args.pv_source_sample_num,
    )


def load_experiment_resources(args):
    model_name = get_model_key(args.model)
    system_prompts = load_system_prompts(args.domain_dir)
    if args.role_name not in system_prompts:
        raise ValueError(f"No system prompt found for role={args.role_name!r}")

    pv_layers_hidden_vector = load_pv_hidden_vector(
        data_dir=build_pv_vector_dir(args),
    )
    model, tokenizer_or_processor = load_model(args.model, args.device, args.dtype)
    return ExperimentResources(
        model_name=model_name,
        model=model,
        tokenizer_or_processor=tokenizer_or_processor,
        system_prompts=system_prompts,
        pv_layers_hidden_vector=pv_layers_hidden_vector,
    )


def resolve_key_layers(args, resources, multimodal=False):
    if not getattr(args, "locate_key_layers", False):
        return get_key_layers(
            resources.model_name,
            multimodal=multimodal,
        )

    from pv_detector.locate_pv_aware_layers import (
        build_runtime_locate_args,
        locate_pv_aware_key_layers,
    )

    modality = "vl" if multimodal else "text"
    locate_args = build_runtime_locate_args(args, modality=modality)
    # Reuse the loaded model and PV vectors during layer localization.
    key_layers = locate_pv_aware_key_layers(
        locate_args,
        model_name=resources.model_name,
        model=resources.model,
        tokenizer=resources.tokenizer_or_processor,
        system_prompts=resources.system_prompts,
        pv_layers_hidden_vector=resources.pv_layers_hidden_vector,
    )
    if not key_layers:
        raise ValueError("Dynamic key-layer localization returned no key layers")
    return list(key_layers)


def prepare_experiment(args, multimodal=False):
    set_seed(args.seed)
    resources = load_experiment_resources(args)
    key_layers = resolve_key_layers(args, resources, multimodal=multimodal)
    # Build output paths only after the actual key layers have been resolved.
    context = build_context(args, key_layers, multimodal=multimodal)
    set_logging(context.log_dir, stdout=True, add_time=True)
    if getattr(args, "locate_key_layers", False):
        logging.info(f"Using dynamically located key layers: {key_layers}")
    detector = make_detector(resources, multimodal=multimodal)
    return context, resources.system_prompts, detector


def is_normal_dataset_name(dataset_name):
    return "normal" in dataset_name.lower() or "ID" in dataset_name


def update_fpr_fnr_results(results, dataset_name, true_labels, detect_results):
    # Report FPR for benign sets and FNR for attack sets; -1 means not applicable.
    if is_normal_dataset_name(dataset_name):
        fpr = evaluate_FPR(true_labels, detect_results)
        logging.info(f"Test {dataset_name}, FPR={fpr}")
        results[dataset_name] = (fpr, -1)
    else:
        fnr = evaluate_FNR(true_labels, detect_results)
        logging.info(f"Test {dataset_name}, FNR={fnr}")
        results[dataset_name] = (-1, fnr)


def make_detector(resources, multimodal=False):
    from pv_detector.PVDetector import PVDetector

    return PVDetector(
        resources.model,
        resources.tokenizer_or_processor,
        model_name=resources.model_name,
        modality="vl" if multimodal else "text",
        pv_layers_hidden_vector=resources.pv_layers_hidden_vector,
    )


def run_custom_text_evaluation(args, context, detector, system_prompts):
    role = args.role_name
    key_layers = context.key_layers
    datasets_config = load_config(args.custom_dataset_config_path.format(role=role))[context.model_name]
    datasets = load_datasets_samples(datasets_config[args.dataset_name])
    logging.info(f"Starting evaluation of {len(datasets)} datasets_config for Custom")

    # Load benign data separately for threshold calibration and FPR evaluation.
    normal_config = get_normal_config(datasets_config)
    datasets.update(load_datasets_samples(normal_config))
    datasets = combine_normal_data(datasets)

    # Recalibrate for the current vectors and key layers on every evaluation run.
    logging.info("Estimating threshold from normal samples at target FPR=%.2f%%", args.fpr_target * 100)
    detector.threshold = detector.estimate_threshold_from_normal(
        datasets["Normal"],
        system_prompt=system_prompts[role],
        key_layers=key_layers,
        projection_type="proj",
        fpr_target=args.fpr_target,
        n_samples=100,
    )

    return evaluate_flat_datasets(detector, datasets, system_prompts[role], key_layers)


def evaluate_flat_datasets(detector, datasets, system_prompt, key_layers):
    results = {}
    for idx, (dataset_name, dataset) in enumerate(datasets.items(), 1):
        logging.info(f"Processing dataset {idx}/{len(datasets)}: {dataset_name}")
        true_labels, detect_results = detector.detect_batch(
            dataset,
            system_prompt,
            key_layers=key_layers,
            projection_type="proj",
        )
        update_fpr_fnr_results(results, dataset_name, true_labels, detect_results)
    return results


def run_text_experiment(args):
    context, system_prompts, detector = prepare_experiment(
        args,
        multimodal=False,
    )

    results = run_custom_text_evaluation(args, context, detector, system_prompts)

    torch.cuda.empty_cache()
    save_results_csv(results, output_path=os.path.join(context.output_dir, f"fpr_fnr_result_{context.timestamp}.csv"))
    return context


def run_vl_experiment(args):
    # VL shares the detector and metrics; each sample also carries its image path.
    context, system_prompts, detector = prepare_experiment(
        args,
        multimodal=True,
    )
    role = args.role_name
    key_layers = context.key_layers

    datasets_config = load_config(args.custom_dataset_config_path.format(role=role))[context.model_name]
    datasets = load_VL_datasets_samples(datasets_config[args.dataset_name])
    logging.info(f"Starting evaluation of {len(datasets)} datasets_config for Custom")

    normal_config = get_normal_config(datasets_config)
    datasets.update(load_VL_datasets_samples(normal_config))
    datasets = combine_normal_data(datasets)

    # VL uses the same per-run calibration policy as text evaluation.
    logging.info("Estimating threshold from normal samples at target FPR=%.2f%%", args.fpr_target * 100)
    detector.threshold = detector.estimate_threshold_from_normal(
        datasets["Normal"],
        system_prompt=system_prompts[role],
        key_layers=key_layers,
        projection_type="proj",
        fpr_target=args.fpr_target,
        n_samples=100,
    )

    results = evaluate_flat_datasets(detector, datasets, system_prompts[role], key_layers)
    torch.cuda.empty_cache()
    save_results_csv(results, output_path=os.path.join(context.output_dir, f"fpr_fnr_result_{context.timestamp}.csv"))
    return context

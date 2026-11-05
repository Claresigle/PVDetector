import argparse
import gc
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
from tqdm import tqdm

from model_config import (
    DEFAULT_PHI_VL_MODEL_ID,
    DEFAULT_TEXT_MODEL_ID,
    MODEL_ARGUMENT_HELP,
    get_model_key,
)

from .dataset import (
    DATASET_CONFIGS,
    build_dataset_paths,
    build_dataset_paths_VL,
    load_json_file,
    load_samples_from_dataset,
    load_samples_from_dataset_VL,
    load_system_prompts,
    save_json_file,
)
from .model import generate_vl_response, llm_generate, load_model
from .other_utils import ATTACK_DIR, REPO_DIR, create_attack_msg, set_logging, set_seed


# The paper's heuristic-based attacks construct inputs from fixed instruction templates.
MANUAL_ATTACK_STRATEGIES = ("Combine", "Ignore", "Fake_Complete")


@dataclass
class TextManualAttackContext:
    args: argparse.Namespace
    model_name: str
    role: str
    model: Any
    tokenizer: Any
    system_prompt: str
    refusal_prefixes: List[str]
    log_path_no_deny: str


@dataclass
class VLManualAttackContext:
    args: argparse.Namespace
    model_name: str
    role: str
    model: Any
    processor: Any
    system_prompt: str
    refusal_prefixes: List[str]
    log_path_no_deny: str


@dataclass
class ManualAttackData:
    samples: List[Any]
    save_path_no_deny: str
    save_path: str


def _build_manual_attack_parser(
        default_model: str,
        default_role: str,
        default_dataset: str,
        default_sample_num: Optional[int],
        default_device: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=default_model, help=MODEL_ARGUMENT_HELP)
    parser.add_argument("--domain_dir", type=str, default=str(REPO_DIR / "datasets" / "Agents_system_prompt"))
    parser.add_argument("--role_name", type=str, default=default_role)
    parser.add_argument("--tar_data_dir", type=str, default=str(REPO_DIR / "datasets"))
    parser.add_argument("--refusal_prefixes_path", type=str, default=str(ATTACK_DIR / "assets" / "refusal_prefixes.json"))
    parser.add_argument(
        "--attack_strategy",
        type=str,
        choices=MANUAL_ATTACK_STRATEGIES,
        default="Combine",
    )
    parser.add_argument("--dataset", type=str, choices=tuple(DATASET_CONFIGS), default=default_dataset)
    parser.add_argument(
        "--sample_num", type=int, default=default_sample_num,
        help="Number of attack samples; VL defaults to all available rejected samples.",
    )
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--log_dir", type=str, default="./logs/manual_attack")
    parser.add_argument("--save_dir", type=str, default=str(REPO_DIR / "datasets" / "Attack_Data"))
    return parser


def build_text_manual_attack_parser(
        default_role: str = "LineMonitor",
        default_dataset: str = "Unauthorized_Action"):
    return _build_manual_attack_parser(
        default_model=DEFAULT_TEXT_MODEL_ID,
        default_role=default_role,
        default_dataset=default_dataset,
        default_sample_num=100,
        default_device="cuda:0",
    )


def build_vl_manual_attack_parser(
        default_role: str = "RecipeMaster",
        default_dataset: str = "Alpaca"):
    return _build_manual_attack_parser(
        default_model=DEFAULT_PHI_VL_MODEL_ID,
        default_role=default_role,
        default_dataset=default_dataset,
        default_sample_num=None,
        default_device="cuda:1",
    )


def _prepare_manual_attack(args):
    model_name = get_model_key(args.model)
    role = args.role_name
    set_seed(args.seed)

    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"{args.dataset} is not supported.")

    log_dir = os.path.join(
        args.log_dir,
        args.attack_strategy,
        role,
        model_name,
        f"{args.dataset}_{args.sample_num}",
    )
    log_path = os.path.join(log_dir, "all")
    log_path_no_deny = os.path.join(log_dir, "no_deny.log")
    set_logging(log_path, stdout=False, add_time=False)
    os.makedirs(log_dir, exist_ok=True)

    model, tokenizer_or_processor = load_model(args.model, args.device, args.dtype)
    system_prompts = load_system_prompts(args.domain_dir)
    if role not in system_prompts:
        raise ValueError(f"System prompt for role '{role}' was not found in {args.domain_dir}.")
    refusal_prefixes = load_json_file(args.refusal_prefixes_path)
    logging.info(
        "Initialized manual attack: strategy=%s, role=%s, model=%s, dataset=%s",
        args.attack_strategy,
        role,
        model_name,
        args.dataset,
    )
    return (
        model_name,
        role,
        model,
        tokenizer_or_processor,
        system_prompts[role],
        refusal_prefixes,
        log_path_no_deny,
    )


def prepare_text_manual_attack_context(args):
    (
        model_name,
        role,
        model,
        tokenizer,
        system_prompt,
        refusal_prefixes,
        log_path_no_deny,
    ) = _prepare_manual_attack(args)
    return TextManualAttackContext(
        args=args,
        model_name=model_name,
        role=role,
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        refusal_prefixes=refusal_prefixes,
        log_path_no_deny=log_path_no_deny,
    )


def prepare_vl_manual_attack_context(args):
    # Resolve the sample count before building log paths and output filenames.
    if args.sample_num is None or args.sample_num == -1:
        model_name = get_model_key(args.model)
        data_path, _, _, _ = build_dataset_paths_VL(
            args, args.role_name, model_name, attack_type=args.attack_strategy,
        )
        if data_path is None:
            raise FileNotFoundError(
                f"No rejected VL dataset found for role={args.role_name}, model={model_name}."
            )
        args.sample_num = len(load_json_file(data_path))
        if args.sample_num == 0:
            raise ValueError(f"Rejected VL samples must not be empty: {data_path}")
    (
        model_name,
        role,
        model,
        processor,
        system_prompt,
        refusal_prefixes,
        log_path_no_deny,
    ) = _prepare_manual_attack(args)
    return VLManualAttackContext(
        args=args,
        model_name=model_name,
        role=role,
        model=model,
        processor=processor,
        system_prompt=system_prompt,
        refusal_prefixes=refusal_prefixes,
        log_path_no_deny=log_path_no_deny,
    )


def _load_manual_attack_data(
        args,
        role: str,
        model_name: str,
        path_builder: Callable,
        sample_loader: Callable):
    data_path, save_dir, save_path_no_deny, save_path = path_builder(
        args,
        role,
        model_name,
        attack_type=args.attack_strategy,
    )
    if not data_path:
        raise FileNotFoundError(
            f"No rejected dataset file found for role={role}, model={model_name}, "
            f"dataset={args.dataset}."
        )

    samples, _ = sample_loader(data_path, sample_num=args.sample_num)
    os.makedirs(save_dir, exist_ok=True)
    return ManualAttackData(
        samples=samples,
        save_path_no_deny=save_path_no_deny,
        save_path=save_path,
    )


def load_text_manual_attack_data(args, role: str, model_name: str):
    return _load_manual_attack_data(
        args,
        role,
        model_name,
        path_builder=build_dataset_paths,
        sample_loader=load_samples_from_dataset,
    )


def load_vl_manual_attack_data(args, role: str, model_name: str):
    return _load_manual_attack_data(
        args,
        role,
        model_name,
        path_builder=build_dataset_paths_VL,
        sample_loader=load_samples_from_dataset_VL,
    )


def _load_existing_manual_results(
        attack_data: ManualAttackData,
        attack_strategy: str,
        get_message: Callable[[Any], str],
        build_result: Callable[[str, Any], Any]):
    attack_result_msgs = []
    attack_success_msgs = []

    if os.path.exists(attack_data.save_path):
        attack_result_msgs = load_json_file(attack_data.save_path)
    if os.path.exists(attack_data.save_path_no_deny):
        attack_success_msgs = load_json_file(attack_data.save_path_no_deny)

    if len(attack_result_msgs) > len(attack_data.samples):
        raise ValueError(
            f"Existing result count ({len(attack_result_msgs)}) exceeds the current sample count "
            f"({len(attack_data.samples)}): {attack_data.save_path}"
        )

    # Validate inputs, strategy, and image paths against saved records before resuming.
    expected_results = [
        build_result(create_attack_msg(get_message(sample), attack_strategy), sample)
        for sample in attack_data.samples[:len(attack_result_msgs)]
    ]
    if attack_result_msgs != expected_results:
        raise ValueError(
            "Existing manual-attack results do not match the current inputs, strategy, or image paths; "
            f"refusing an unsafe resume from {attack_data.save_path}."
        )

    # Match an ordered subsequence so duplicate records consume separate occurrences.
    committed_success_msgs = []
    result_index = 0
    for message in attack_success_msgs:
        while result_index < len(attack_result_msgs) and attack_result_msgs[result_index] != message:
            result_index += 1
        if result_index == len(attack_result_msgs):
            break
        committed_success_msgs.append(message)
        result_index += 1

    # The success file may be one sample ahead if its write completed before a crash.
    if len(committed_success_msgs) != len(attack_success_msgs):
        logging.warning(
            "Discarding %d uncommitted successful manual-attack result(s) from %s",
            len(attack_success_msgs) - len(committed_success_msgs),
            attack_data.save_path_no_deny,
        )
        attack_success_msgs = committed_success_msgs

    if attack_result_msgs:
        logging.info(
            "Loaded %d completed manual-attack results from %s",
            len(attack_result_msgs),
            attack_data.save_path,
        )
    else:
        logging.info("No existing manual-attack results found; starting a new run.")
    return attack_success_msgs, attack_result_msgs


def _save_manual_results(
        attack_data: ManualAttackData,
        attack_success_msgs: List[Any],
        attack_result_msgs: List[Any]):
    # The all-results file is the commit marker; write it after the success subset.
    save_json_file(attack_data.save_path_no_deny, attack_success_msgs)
    save_json_file(attack_data.save_path, attack_result_msgs)


def _log_attack_result(
        sample_idx: int,
        injected_message: str,
        generation: str,
        log_path_no_deny: str,
        is_success: bool):
    logging.info(f"==========Inject_attack_msg[{sample_idx}]===========")
    logging.info(f"Prompt:\n{injected_message}\n")
    logging.info(f"Generation:\n{generation}\n\n")
    if is_success:
        with open(log_path_no_deny, "a", encoding="utf-8") as log_file:
            log_file.write(f"==========Inject_attack_msg[{sample_idx}]===========\n")
            log_file.write(f"Prompt:\n{injected_message}\n\n")
            log_file.write(f"Generation:\n{generation}\n\n")


def _clear_runtime_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _run_manual_attack(
        context,
        attack_data: ManualAttackData,
        get_message: Callable[[Any], str],
        generate_response: Callable[[str, Any], str],
        modality_label: str,
        build_result: Callable[[str, Any], Any]):
    # Modality callbacks supply text extraction, generation, and output formatting.
    attack_success_msgs, attack_result_msgs = _load_existing_manual_results(
        attack_data,
        context.args.attack_strategy,
        get_message,
        build_result,
    )
    completed_num = len(attack_result_msgs)
    if completed_num == 0 and os.path.exists(context.log_path_no_deny):
        os.remove(context.log_path_no_deny)
    # Persist checkpoint repairs before another generation can fail.
    _save_manual_results(attack_data, attack_success_msgs, attack_result_msgs)

    progress = tqdm(
        range(completed_num, len(attack_data.samples)),
        initial=completed_num,
        total=len(attack_data.samples),
        desc=(
            f"Running {context.args.attack_strategy} manual attack on "
            f"{context.args.dataset} {modality_label}dataset"
        ),
    )
    for sample_idx in progress:
        sample = attack_data.samples[sample_idx]
        injected_message = create_attack_msg(
            get_message(sample),
            context.args.attack_strategy,
        )
        try:
            generation = generate_response(injected_message, sample)
            # A missing refusal prefix is the heuristic criterion for attack success.
            is_success = not generation.startswith(tuple(context.refusal_prefixes))
            result = build_result(injected_message, sample)
            attack_result_msgs.append(result)
            if is_success:
                attack_success_msgs.append(result)
            # Save every completed sample; the all-results count determines resume progress.
            _save_manual_results(
                attack_data,
                attack_success_msgs,
                attack_result_msgs,
            )
            _log_attack_result(
                sample_idx,
                injected_message,
                generation,
                context.log_path_no_deny,
                is_success,
            )
        finally:
            _clear_runtime_cache()

    _save_manual_results(
        attack_data,
        attack_success_msgs,
        attack_result_msgs,
    )


def run_text_manual_attack(
        context: TextManualAttackContext,
        attack_data: ManualAttackData):
    def generate_response(injected_message, _sample):
        return llm_generate(
            context.model,
            context.model_name,
            context.tokenizer,
            context.system_prompt,
            injected_message,
            context.args.device,
        )

    _run_manual_attack(
        context,
        attack_data,
        get_message=lambda sample: sample,
        generate_response=generate_response,
        modality_label="",
        build_result=lambda message, _sample: message,
    )


def run_vl_manual_attack(
        context: VLManualAttackContext,
        attack_data: ManualAttackData):
    # Save attack text with its original image path to preserve sampled pairs.
    def generate_response(injected_message, sample):
        _, img_path = sample
        return generate_vl_response(
            context.model_name,
            context.model,
            context.processor,
            context.system_prompt,
            injected_message,
            img_path,
            context.args.device,
        )

    _run_manual_attack(
        context,
        attack_data,
        get_message=lambda sample: sample[0],
        generate_response=generate_response,
        modality_label="VL ",
        build_result=lambda message, sample: {"msg": message, "img_path": sample[1]},
    )


def execute_text_manual_attack(args):
    context = prepare_text_manual_attack_context(args)
    attack_data = load_text_manual_attack_data(
        args,
        context.role,
        context.model_name,
    )
    run_text_manual_attack(
        context=context,
        attack_data=attack_data,
    )


def execute_vl_manual_attack(args):
    context = prepare_vl_manual_attack_context(args)
    attack_data = load_vl_manual_attack_data(
        args,
        context.role,
        context.model_name,
    )
    run_vl_manual_attack(
        context=context,
        attack_data=attack_data,
    )

import argparse
import gc
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from model_config import (
    DEFAULT_TEXT_MODEL_ID,
    DEFAULT_VL_MODEL_ID,
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
from .model import generate_vl_response, llm_generate, load_model, prepare_vl_model
from .other_utils import ATTACK_DIR, REPO_DIR, set_logging, set_seed


@dataclass
class TextGCGContext:
    args: argparse.Namespace
    model_name: str
    role: str
    model: object
    tokenizer: object
    system_prompt: str
    refusal_prefixes: List[str]
    log_path_no_deny: str


@dataclass
class TextAttackData:
    samples: List[str]
    target: str
    save_path_no_deny: str
    save_path: str


@dataclass
class VLGCGContext:
    args: argparse.Namespace
    model_name: str
    role: str
    model: Any
    processor: Any
    tokenizer: Any
    system_prompt: str
    refusal_prefixes: List[str]
    log_path_no_deny: str


@dataclass
class VLAttackData:
    samples: List[tuple]
    target: str
    save_dir: str
    save_img_dir: str
    save_path_no_deny: str
    save_path: str


def build_text_gcg_parser(
        default_role: str,
        default_dataset: str,
        default_log_dir: str = "./logs/gcg",
        default_model: str = DEFAULT_TEXT_MODEL_ID,
        default_sample_num: int = 100,
        default_device: str = "cuda:0",
        default_save_dir: str = str(REPO_DIR / "datasets" / "Attack_Data")):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=default_model, help=MODEL_ARGUMENT_HELP)
    parser.add_argument("--domain_dir", type=str, default=str(REPO_DIR / "datasets" / "Agents_system_prompt"))
    parser.add_argument("--role_name", type=str, default=default_role)
    parser.add_argument("--tar_data_dir", type=str, default=str(REPO_DIR / "datasets"))
    parser.add_argument("--refusal_prefixes_path", type=str, default=str(ATTACK_DIR / "assets" / "refusal_prefixes.json"))
    parser.add_argument("--dataset", default=default_dataset, type=str)
    parser.add_argument("--sample_num", type=int, default=default_sample_num)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--log_dir", type=str, default=default_log_dir)
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    return parser


def build_text_gcg_config(config_cls, args):
    return config_cls(
        num_steps=300,
        search_width=256,
        topk=256,
        batch_size=512,
        seed=args.seed,
        verbosity="INFO",
        probe_sampling_config=None,
    )


def build_vl_gcg_parser(
        default_role: str = "RecipeMaster",
        default_dataset: str = "Alpaca",
        default_sample_num: int = 50):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VL_MODEL_ID,
        help=MODEL_ARGUMENT_HELP,
    )
    parser.add_argument("--domain_dir", type=str, default=str(REPO_DIR / "datasets" / "Agents_system_prompt"))
    parser.add_argument("--role_name", type=str, default=default_role)
    parser.add_argument("--tar_data_dir", type=str, default=str(REPO_DIR / "datasets"))
    parser.add_argument("--refusal_prefixes_path", type=str, default=str(ATTACK_DIR / "assets" / "refusal_prefixes.json"))
    parser.add_argument("--dataset", default=default_dataset, type=str)
    parser.add_argument("--sample_num", type=int, default=default_sample_num)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--log_dir", type=str, default="./logs/gcg")
    parser.add_argument("--save_dir", type=str, default=str(REPO_DIR / "datasets" / "Attack_Data"))
    parser.add_argument("--target_keyword", type=str, default="I can help with that")
    return parser


def load_vl_gcg_backend(model_name):
    # Each VLM backend handles its own visual tokens and input format.
    if "Qwen2.5" in model_name:
        from ..nanogcg_qwen_vl import GCGConfig
        from ..nanogcg_qwen_vl.gcg import GCG

        return GCGConfig, GCG
    if "Phi" in model_name:
        from ..nanogcg_phi_vl import GCGConfig
        from ..nanogcg_phi_vl.gcg import GCG

        return GCGConfig, GCG
    raise ValueError(f"{model_name} is not supported.")


def build_vl_gcg_config(config_cls, args):
    return config_cls(
        num_steps=100,
        search_width=32,
        topk=64,
        batch_size=32,
        seed=args.seed,
        verbosity="INFO",
    )


def build_vl_prompt_template(model_name, processor, system_prompt, msg, img_path, prompt_prefix: str = ""):
    # The backend replaces this placeholder with the suffix being optimized.
    user_text = f"{prompt_prefix}{msg}\n" + "{optim_str}"
    if "Qwen2.5" in model_name:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if "Phi" in model_name:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "<|image_1|>\n" + user_text,
            },
        ]
        return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    raise ValueError(f"{model_name} is not supported.")


def prepare_text_gcg_context(args, attack_name: str):
    model_name = get_model_key(args.model)
    role = args.role_name
    set_seed(args.seed)

    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"{args.dataset} is not supported.")

    run_name = f"{args.dataset}_{args.sample_num}"
    log_path = f"{args.log_dir}/{role}/{model_name}/{run_name}/all"
    set_logging(log_path, stdout=False, add_time=False)
    log_path_no_deny = f"{os.path.dirname(log_path)}/no_deny.log"

    model, tokenizer = load_model(args.model, args.device, args.dtype)
    system_prompts = load_system_prompts(args.domain_dir)
    if role not in system_prompts:
        raise ValueError(f"System prompt for role '{role}' was not found in {args.domain_dir}.")
    refusal_prefixes = load_json_file(args.refusal_prefixes_path)
    logging.info(f"Initialized {attack_name}: role={role}, model={model_name}, dataset={args.dataset}")

    return TextGCGContext(
        args=args,
        model_name=model_name,
        role=role,
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompts[role],
        refusal_prefixes=refusal_prefixes,
        log_path_no_deny=log_path_no_deny,
    )


def prepare_vl_gcg_context(args, attack_name: str):
    model_name = get_model_key(args.model)
    role = args.role_name
    set_seed(args.seed)

    if args.dataset != "Alpaca":
        raise ValueError(f"{args.dataset} is not supported.")

    log_path = f"{args.log_dir}/{role}/{model_name}/{args.dataset}_{args.sample_num}/all"
    set_logging(log_path, stdout=False, add_time=False)
    log_path_no_deny = f"{os.path.dirname(log_path)}/no_deny.log"
    if os.path.exists(log_path_no_deny):
        os.remove(log_path_no_deny)

    model, processor = load_model(args.model, args.device, args.dtype)
    tokenizer = prepare_vl_model(model, processor)
    system_prompts = load_system_prompts(args.domain_dir)
    if role not in system_prompts:
        raise ValueError(f"System prompt for role '{role}' was not found in {args.domain_dir}.")
    refusal_prefixes = load_json_file(args.refusal_prefixes_path)
    logging.info(f"Initialized {attack_name}: role={role}, model={model_name}, dataset={args.dataset}")

    return VLGCGContext(
        args=args,
        model_name=model_name,
        role=role,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        system_prompt=system_prompts[role],
        refusal_prefixes=refusal_prefixes,
        log_path_no_deny=log_path_no_deny,
    )


def load_text_attack_data(
        args,
        role: str,
        model_name: str,
        attack_type: str):
    data_path, save_dir, save_path_no_deny, save_path = build_dataset_paths(
        args,
        role,
        model_name,
        attack_type=attack_type,
    )
    if not data_path:
        raise FileNotFoundError(
            f"No rejected dataset file found for role={role}, model={model_name}, dataset={args.dataset}."
        )

    samples, target = load_samples_from_dataset(data_path, sample_num=args.sample_num)
    os.makedirs(save_dir, exist_ok=True)
    return TextAttackData(
        samples=samples,
        target=target,
        save_path_no_deny=save_path_no_deny,
        save_path=save_path,
    )


def load_vl_attack_data(args, role: str, model_name: str, attack_type: str):
    data_path, save_dir, save_path_no_deny, save_path = build_dataset_paths_VL(
        args,
        role,
        model_name,
        attack_type=attack_type,
    )
    if not data_path:
        raise FileNotFoundError(
            f"No rejected dataset file found for role={role}, model={model_name}, dataset={args.dataset}."
        )

    samples, target = load_samples_from_dataset_VL(data_path, sample_num=args.sample_num)
    save_img_dir = os.path.join(
        save_dir,
        f"{DATASET_CONFIGS[args.dataset]['output_prefix']}_{args.sample_num}it_images",
    )
    os.makedirs(save_img_dir, exist_ok=True)
    return VLAttackData(
        samples=samples,
        target=target,
        save_dir=save_dir,
        save_img_dir=save_img_dir,
        save_path_no_deny=save_path_no_deny,
        save_path=save_path,
    )


def load_existing_text_results(
        attack_data: TextAttackData,
        attack_name: str,
        prompt_prefix: str = ""):
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

    # Check saved input prefixes to avoid resuming with a different sample order.
    for sample_idx, (sample, result) in enumerate(zip(attack_data.samples, attack_result_msgs)):
        expected_prefix = f"{prompt_prefix}{sample}\n"
        if not result.startswith(expected_prefix):
            raise ValueError(
                "Existing text-GCG results do not match the current sampled inputs; "
                f"refusing an unsafe resume at sample {sample_idx}: {attack_data.save_path}"
            )

    # A crash between the two file writes may leave success entries uncommitted.
    committed_success_msgs = [
        message for message in attack_success_msgs
        if message in attack_result_msgs
    ]
    if len(committed_success_msgs) != len(attack_success_msgs):
        logging.warning(
            "Discarding %d uncommitted successful text-GCG result(s) from %s",
            len(attack_success_msgs) - len(committed_success_msgs),
            attack_data.save_path_no_deny,
        )
        attack_success_msgs = committed_success_msgs

    if attack_result_msgs:
        logging.info(f"Loaded {len(attack_result_msgs)} attack results from {attack_data.save_path}")
    else:
        logging.info(f"No existing {attack_name} results found; starting a new run.")

    remaining_samples = attack_data.samples[len(attack_result_msgs):]
    return attack_success_msgs, attack_result_msgs, remaining_samples


def save_text_gcg_results(
        attack_data: TextAttackData,
        attack_success_msgs: List[str],
        attack_result_msgs: List[str]):
    # The all-results file is the completion marker. Save it last so an
    # interrupted two-file checkpoint never skips an uncommitted sample.
    save_json_file(attack_data.save_path_no_deny, attack_success_msgs)
    save_json_file(attack_data.save_path, attack_result_msgs)


def load_existing_vl_results(attack_data: VLAttackData, attack_name: str):
    attack_success_msgs = []
    attack_result_msgs = []
    remaining_samples = attack_data.samples

    if os.path.exists(attack_data.save_path) and os.path.exists(attack_data.save_path_no_deny):
        attack_result_msgs = load_json_file(attack_data.save_path)
        attack_success_msgs = load_json_file(attack_data.save_path_no_deny)
        logging.info(f"Loaded {len(attack_result_msgs)} attack results from {attack_data.save_path}")
        remaining_samples = attack_data.samples[len(attack_result_msgs):]
    else:
        logging.info(f"No existing {attack_name} results found; starting a new VL run.")

    return attack_success_msgs, attack_result_msgs, remaining_samples


def run_text_gcg_attack(
        context: TextGCGContext,
        attack_data: TextAttackData,
        gcg_config,
        run_fn: Callable,
        attack_name: str,
        prompt_prefix: str = ""):
    
    attack_success_msgs, attack_result_msgs, remaining_samples = load_existing_text_results(
        attack_data,
        attack_name,
        prompt_prefix=prompt_prefix,
    )
    completed_num = len(attack_result_msgs)
    if completed_num == 0 and os.path.exists(context.log_path_no_deny):
        os.remove(context.log_path_no_deny)
    save_text_gcg_results(attack_data, attack_success_msgs, attack_result_msgs)

    for sample_idx in tqdm(
            range(completed_num, len(attack_data.samples)),
            initial=completed_num,
            total=len(attack_data.samples),
            desc=f"Running {attack_name} in {context.args.dataset} dataset"):
        att_data = attack_data.samples[sample_idx]
        user_content = f"{prompt_prefix}{att_data}\n"
        message = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            # Optimize the suffix, then generate a reply to assess the resulting input.
            result = run_fn(
                context.model,
                context.tokenizer,
                message,
                attack_data.target,
                gcg_config,
            )
            injected_message = user_content + result.best_string
            generation = llm_generate(
                context.model,
                context.model_name,
                context.tokenizer,
                context.system_prompt,
                injected_message,
                context.args.device,
            )
            is_success = not generation.startswith(tuple(context.refusal_prefixes))
            attack_result_msgs.append(injected_message)
            if is_success:
                attack_success_msgs.append(injected_message)
            # Commit each sample before proceeding to the next optimization.
            save_text_gcg_results(
                attack_data,
                attack_success_msgs,
                attack_result_msgs,
            )

            logging.info(f"==========Inject_attack_msg[{sample_idx}]===========")
            logging.info(f"Prompt:\n{injected_message}\n")
            logging.info(f"Generation:\n{generation}\n\n")
            if is_success:
                with open(context.log_path_no_deny, "a", encoding="utf-8") as f:
                    f.write(f"==========Inject_attack_msg[{sample_idx}]===========\n")
                    f.write(f"Prompt:\n{injected_message}\n\n")
                    f.write(f"Generation:\n{generation}\n\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    save_text_gcg_results(attack_data, attack_success_msgs, attack_result_msgs)


def run_vl_gcg_attack(
        context: VLGCGContext,
        attack_data: VLAttackData,
        gcg_attacker,
        attack_name: str = "GCG",
        prompt_prefix: str = ""):
    attack_success_msgs, attack_result_msgs, remaining_samples = load_existing_vl_results(
        attack_data,
        attack_name,
    )
    completed_num = len(attack_result_msgs)

    for offset, (msg, img_path) in tqdm(
            enumerate(remaining_samples),
            total=len(remaining_samples),
            desc=f"Running {attack_name} in {context.args.dataset} VL dataset"):
        sample_idx = completed_num + offset
        torch.cuda.empty_cache()
        gc.collect()

        # Copy images in output order to match the saved list of attack strings.
        img_new_path = os.path.join(attack_data.save_img_dir, f"img_{sample_idx + 1}.jpg")
        shutil.copy2(img_path, img_new_path)
        pil_image = Image.open(img_path).convert("RGB")
        full_prompt_str = build_vl_prompt_template(
            context.model_name,
            context.processor,
            context.system_prompt,
            msg,
            img_path,
            prompt_prefix=prompt_prefix,
        )

        result = gcg_attacker.run(
            prompt_template=full_prompt_str,
            target=attack_data.target,
            image=pil_image,
            target_keyword=context.args.target_keyword,
        )

        injected_msg = f"{prompt_prefix}{msg}\n" + result.best_string
        attack_result_msgs.append(injected_msg)
        save_json_file(attack_data.save_path, attack_result_msgs)
        generation = generate_vl_response(
            context.model_name,
            context.model,
            context.processor,
            context.system_prompt,
            injected_msg,
            img_path,
            context.args.device,
        )
        logging.info(f"==========Inject_attack_msg[{sample_idx}]===========")
        logging.info(f"Prompt:\n{injected_msg}\n")
        logging.info(f"Generation:\n{generation}\n\n")

        if not generation.startswith(tuple(context.refusal_prefixes)):
            attack_success_msgs.append(injected_msg)
            save_json_file(attack_data.save_path_no_deny, attack_success_msgs)
            with open(context.log_path_no_deny, "a", encoding="utf-8") as f:
                f.write(f"==========Inject_attack_msg[{sample_idx}]===========\n")
                f.write(f"Prompt:\n{injected_msg}\n\n")
                f.write(f"Generation:\n{generation}\n\n")

    save_json_file(attack_data.save_path_no_deny, attack_success_msgs)
    save_json_file(attack_data.save_path, attack_result_msgs)


def execute_text_gcg(
        args,
        attack_name: str,
        prompt_prefix: str = "",
        storage_attack_name: Optional[str] = None):
    from .. import nanogcg
    from ..nanogcg import GCGConfig

    context = prepare_text_gcg_context(args, attack_name=attack_name)
    gcg_config = build_text_gcg_config(GCGConfig, args)
    attack_data = load_text_attack_data(
        args,
        context.role,
        context.model_name,
        attack_type=storage_attack_name or attack_name,
    )
    run_text_gcg_attack(
        context=context,
        attack_data=attack_data,
        gcg_config=gcg_config,
        run_fn=nanogcg.run,
        attack_name=attack_name,
        prompt_prefix=prompt_prefix,
    )


def execute_vl_gcg(
        args,
        attack_name: str,
        prompt_prefix: str = "",
        storage_attack_name: Optional[str] = None):
    context = prepare_vl_gcg_context(args, attack_name=attack_name)
    config_cls, attacker_cls = load_vl_gcg_backend(context.model_name)
    gcg_config = build_vl_gcg_config(config_cls, args)
    gcg_attacker = attacker_cls(
        context.model,
        context.tokenizer,
        gcg_config,
        context.processor,
    )
    attack_data = load_vl_attack_data(
        args,
        context.role,
        context.model_name,
        attack_type=storage_attack_name or attack_name,
    )
    run_vl_gcg_attack(
        context=context,
        attack_data=attack_data,
        gcg_attacker=gcg_attacker,
        attack_name=attack_name,
        prompt_prefix=prompt_prefix,
    )

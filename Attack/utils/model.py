# -*-coding:utf-8-*-
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


def load_model(model_name_or_path: str, device: str, torch_dtype):
    if "Qwen2.5-VL" in model_name_or_path:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, torch_dtype),
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        return model, processor
    elif "Phi-3.5-vision" in model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=getattr(torch, torch_dtype),
            _attn_implementation='eager',
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            num_crops=4
        )
        return model, processor
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, torch_dtype),
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer


def llm_generate(model, model_name, tokenizer, system_prompt, prompt, device):
    # Construct conversation messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    if "qwen3" in model_name.lower():
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
            return_tensors="pt").to(device)
    else:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt").to(device)
    # for most cases，attention_mask Can simply be set to a mask of non-padding tokens in the input
    attention_mask = (inputs != tokenizer.pad_token_id).long().to(device)
    output = model.generate(inputs,
                            # temperature=0.0,
                            attention_mask=attention_mask,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False, max_new_tokens=128)
    generation = tokenizer.batch_decode(output[:, inputs.shape[1]:], skip_special_tokens=True)[0]
    return generation


def qwen_vl_generate(model, processor, system_prompt, prompt, img_path, device):
    # Construct conversation messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text


def phi_vl_generate(model, processor, system_prompt, prompt, img_path, device):
    # Construct conversation messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "<|image_1|>\n" + f"{prompt}"
        }
    ]
    # Preparation for inference
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=text,
        images=[Image.open(img_path)],
        return_tensors="pt",
    ).to(device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, do_sample=False,
                                   eos_token_id=processor.tokenizer.eos_token_id,
                                   max_new_tokens=128)
    generate_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    return response


def generate_vl_response(model_name, model, processor, system_prompt, prompt, img_path, device):
    if "Qwen2.5" in model_name:
        return qwen_vl_generate(model, processor, system_prompt, prompt, img_path, device)
    if "Phi" in model_name:
        return phi_vl_generate(model, processor, system_prompt, prompt, img_path, device)
    raise ValueError(f"{model_name} is not supported.")


def prepare_vl_model(model, processor):
    # Freeze model weights; GCG differentiates with respect to the suffix inputs.
    model.requires_grad_(False)
    tokenizer = processor.tokenizer
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        print(f"Resizing embeddings: {model.get_input_embeddings().num_embeddings} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        model.get_input_embeddings().weight.requires_grad_(False)
    return tokenizer

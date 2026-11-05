import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


def is_vision_language_model(model_name_or_path):
    model_name = model_name_or_path.lower()
    return "qwen2.5-vl" in model_name or "phi-3.5-vision" in model_name


def load_model(model_name_or_path: str, device: str, torch_dtype):
    normalized_model_name = model_name_or_path.lower()
    if "qwen2.5-vl" in normalized_model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, torch_dtype),
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        return model, processor
    elif "phi-3.5-vision" in normalized_model_name:
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


def get_hidden_states(model, tokenizer, msg, system_prompt):
    """Helper function."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg}
    ]
    if hasattr(tokenizer, "name_or_path"):
        model_name = tokenizer.name_or_path.lower()
    elif hasattr(tokenizer, "config") and hasattr(tokenizer.config, "model_type"):
        model_name = tokenizer.config.model_type.lower()
    else:
        raise RuntimeError("Could not determine model name from tokenizer")
    
    if "qwen3" in model_name or "qwen-3" in model_name:
        text = tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             enable_thinking=False,  # Switches to non-thinking modes.
                                             add_generation_prompt=True)
    else:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    
    input_ids = model_inputs.input_ids  
    attention_mask = model_inputs.attention_mask  

    # Detection needs hidden states from a forward pass, without generating a reply.
    with torch.no_grad():
        
        outputs = model(
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            output_hidden_states=True,  
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Drop the embedding output so index 0 denotes the first Transformer layer.
    return outputs.hidden_states[1:]


def get_hidden_states_qwen_vl(model, processor, msg, img_path, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": msg},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from qwen_vl_utils import process_vision_info
    # Match image positions in the chat template to the actual visual inputs.
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        
        outputs = model(
            **inputs,  
            output_hidden_states=True,  
            output_vision_hidden_states=False,  
        )
    return outputs.hidden_states[1:]


def get_hidden_states_phi_vl(model, processor, msg, img_path, system_prompt):
    
    # Phi uses <|image_1|> to mark the supplied image's position in the chat template.
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content":  f"{msg}\n" + "<|image_1|>\n"
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
    ).to(model.device)
    with torch.no_grad():
        
        outputs = model(
            **inputs,  
            output_hidden_states=True,  
            use_cache=False,
        )
    return outputs.hidden_states[1:]


def get_hidden_states_vl(model, processor, msg, img_path, system_prompt, model_name):
    # Adapt model-specific templates and return layer states without the embedding output.
    normalized_model_name = model_name.lower()
    if "qwen2.5-vl" in normalized_model_name:
        return get_hidden_states_qwen_vl(model, processor, msg, img_path, system_prompt)
    if "phi-3.5-vision" in normalized_model_name:
        return get_hidden_states_phi_vl(model, processor, msg, img_path, system_prompt)
    raise ValueError(f"Unsupported vision-language model: {model_name}")

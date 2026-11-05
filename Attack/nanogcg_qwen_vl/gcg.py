import torch
import logging
from dataclasses import dataclass, field
from typing import List
from torch import Tensor
from tqdm import tqdm
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizer

from .utils import get_nonascii_toks

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GCGConfig:
    num_steps: int = 200
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    search_width: int = 64
    batch_size: int = 64
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 1
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = 42
    verbosity: str = "INFO"
    test_steps: int = 10


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    success: bool
    final_response: str
    # === [Revise] Added new logs field to store intermediate processes ===
    logs: List[dict] = field(default_factory=list)


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return
        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)
        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]


def sample_ids_from_grad(ids, grad, search_width, topk=256, n_replace=1, not_allowed_ids=None):
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)
    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")
    topk_ids = (-grad).topk(topk, dim=1).indices
    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)
    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)
    return new_ids


def filter_ids(ids, tokenizer):
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []
    for i in range(len(ids_decoded)):
        ids_encoded =\
        tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])
    if not filtered_ids:
        return ids
    return torch.stack(filtered_ids)


class GCG:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: GCGConfig, processor=None):
        self.model = model
        self.device = model.device
        self.processor = processor if processor else tokenizer
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        self.config = config
        self.embedding_layer = model.get_input_embeddings()
        self.embedding_device = self.embedding_layer.weight.device
        self.vocab_size = self.embedding_layer.num_embeddings

        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(self.tokenizer, device=self.device)
        self.vision_kwargs = {}

    def _safe_clamp(self, ids, name):
        self.vocab_size = self.embedding_layer.num_embeddings
        if ids.max() >= self.vocab_size or ids.min() < 0:
            ids[ids >= self.vocab_size] = 0
            ids[ids < 0] = 0
        return ids

    def test_generation(self, optim_str: str, prompt_template: str, image: Image.Image) -> str:
        with torch.inference_mode():
            # Qwen processing logic: replace placeholder
            final_prompt = prompt_template.replace('{optim_str}', " " + optim_str)

            
            inputs = self.processor(text=[final_prompt], images=[image], videos=None, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)[0]

            return output_text.strip()

    def run(self, prompt_template: str, target: str, image: Image.Image,
            target_keyword: str
            ) -> GCGResult:
        if '{optim_str}' not in prompt_template:
            raise ValueError("Prompt must contain '{optim_str}'")

        # 1. Split Prompt
        parts = prompt_template.split('{optim_str}')
        before_text = parts[0]
        after_text = parts[1] if len(parts) > 1 else ""

        
        initial_inputs = self.processor(text=[before_text], images=[image], videos=None, padding=True,
                                        return_tensors="pt")
        initial_inputs = initial_inputs.to(self.device)

        self.before_ids = initial_inputs['input_ids'].to(self.embedding_device)

        # 3. Save Vision related parameters
        self.vision_kwargs = {}
        exclude = ['input_ids', 'attention_mask', 'token_type_ids']
        for k, v in initial_inputs.items():
            if k not in exclude:
                if isinstance(v, torch.Tensor):
                    self.vision_kwargs[k] = v.to(self.device)
                else:
                    self.vision_kwargs[k] = v

        
        self.after_ids = torch.tensor([[]], device=self.embedding_device, dtype=torch.long)
        if after_text:
            self.after_ids = self.tokenizer(after_text, add_special_tokens=False, return_tensors="pt")['input_ids'].to(
                self.embedding_device)

        # 5. Process Target
        target_str_proc = " " + target if self.config.add_space_before_target else target
        self.target_ids = self.tokenizer(target_str_proc, add_special_tokens=False, return_tensors="pt")[
            'input_ids'].to(self.embedding_device)

        self.before_ids = self._safe_clamp(self.before_ids, "before_ids")
        self.target_ids = self._safe_clamp(self.target_ids, "target_ids")
        if self.after_ids.shape[1] > 0:
            self.after_ids = self._safe_clamp(self.after_ids, "after_ids")

        # 6. Get Embeddings
        if self.before_ids.shape[1] > 0:
            self.before_embeds = self.embedding_layer(self.before_ids)
        else:
            self.before_embeds = torch.empty((1, 0, self.embedding_layer.weight.shape[1]), device=self.embedding_device)

        if self.after_ids.shape[1] > 0:
            self.after_embeds = self.embedding_layer(self.after_ids)
        else:
            self.after_embeds = torch.empty((1, 0, self.embedding_layer.weight.shape[1]), device=self.embedding_device)

        self.target_embeds = self.embedding_layer(self.target_ids)

        # 7. Initialize Buffer and Loop
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids().to(self.embedding_device)

        losses = []
        optim_strings = []
        success = False
        final_response = ""

        # === [Revise] Initialize log list ===
        attack_logs = []

        pbar = tqdm(range(self.config.num_steps), desc="GCG Steps", leave=False)
        for step_idx in pbar:
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    self.config.search_width,
                    self.config.topk,
                    self.config.n_replace,
                    not_allowed_ids=self.not_allowed_ids
                )
                if self.config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, self.tokenizer)

                sampled_ids = self._safe_clamp(sampled_ids.to(self.embedding_device), "sampled_ids")

                optim_embeds = self.embedding_layer(sampled_ids)

                loss_list = self._compute_candidates_loss(self.config.batch_size, optim_embeds, sampled_ids)

                min_loss = loss_list.min().item()
                best_idx = loss_list.argmin()
                optim_ids = sampled_ids[best_idx].unsqueeze(0)

                losses.append(min_loss)
                buffer.add(min_loss, optim_ids)

            optim_ids = buffer.get_best_ids().to(self.embedding_device)
            optim_str = self.tokenizer.decode(optim_ids[0])
            optim_strings.append(optim_str)
            pbar.set_postfix({'loss': f"{min_loss:.4f}"})

            # === [Revise] Detection logic：Contains step 1，and every test_steps steps ===
            if step_idx == 0 or (step_idx + 1) % self.config.test_steps == 0 or step_idx == self.config.num_steps - 1:
                torch.cuda.empty_cache()
                resp = self.test_generation(optim_str, prompt_template, image)

                # === [Revise] Record the details of the current step ===
                log_entry = {
                    "step": step_idx + 1,
                    "loss": float(min_loss),
                    "suffix": optim_str,
                    "response": resp
                }
                attack_logs.append(log_entry)
                # ================================

                if target_keyword in resp:
                    logger.info(f"Success at step {step_idx}! Keyword '{target_keyword}' found.")
                    success = True
                    final_response = resp
                    break

        # if not success:
        #     final_response = self.test_generation(optim_strings[-1], prompt_template, image)
        #     # If the last step has not been recorded（And failed to exit early），Make a note
        #     if not attack_logs or attack_logs[-1]["step"] != self.config.num_steps:
        #         attack_logs.append({
        #             "step": self.config.num_steps,
        #             "loss": losses[-1] if losses else 0,
        #             "suffix": optim_strings[-1],
        #             "response": final_response
        #         })

        return GCGResult(
            best_loss=losses[-1] if losses else 0,
            best_string=optim_strings[-1] if optim_strings else "",
            losses=losses,
            strings=optim_strings,
            success=success,
            final_response=final_response,
            # === [Revise] Return logs ===
            logs=attack_logs
        )

    def init_buffer(self) -> AttackBuffer:
        buffer = AttackBuffer(self.config.buffer_size)
        init_ids = self.tokenizer(self.config.optim_str_init, add_special_tokens=False, return_tensors="pt")[
            "input_ids"].to(self.embedding_device)
        init_ids = self._safe_clamp(init_ids, "init_ids")
        init_embeds = self.embedding_layer(init_ids)
        loss = self._compute_candidates_loss(1, init_embeds, init_ids)
        buffer.add(loss[0].item(), init_ids)
        return buffer

    def compute_token_gradient(self, optim_ids):
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=self.vocab_size).to(self.embedding_device,
                                                                                                  self.model.dtype)
        optim_ids_onehot.requires_grad_()
        optim_embeds = optim_ids_onehot @ self.embedding_layer.weight

        full_embeds = torch.cat(
            [self.before_embeds.detach(), optim_embeds, self.after_embeds.detach(), self.target_embeds.detach()], dim=1)
        attention_mask = torch.ones((1, full_embeds.shape[1]), device=self.embedding_device, dtype=torch.long)

        # Pass in the vision_kwargs required by Qwen
        outputs = self.model(inputs_embeds=full_embeds, attention_mask=attention_mask, **self.vision_kwargs)

        shift = full_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = outputs.logits[..., shift - 1: -1, :].contiguous()
        shift_labels = self.target_ids.view(-1)

        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels)
        grad = torch.autograd.grad(loss, optim_ids_onehot)[0]
        return grad

    def _compute_candidates_loss(self, batch_size, optim_embeds, optim_ids):
        all_loss = []
        actual_batch_size = max(1, batch_size)

        for i in range(0, optim_embeds.shape[0], actual_batch_size):
            optim_embeds_batch = optim_embeds[i:i + actual_batch_size]
            current_bs = optim_embeds_batch.shape[0]

            before_embeds_batch = self.before_embeds.repeat(current_bs, 1, 1)
            after_embeds_batch = self.after_embeds.repeat(current_bs, 1, 1)
            target_embeds_batch = self.target_embeds.repeat(current_bs, 1, 1)

            before_ids_batch = self.before_ids.repeat(current_bs, 1)
            optim_ids_batch = optim_ids[i:i + actual_batch_size]
            after_ids_batch = self.after_ids.repeat(current_bs, 1)
            target_ids_batch = self.target_ids.repeat(current_bs, 1)

            inputs_embeds = torch.cat(
                [before_embeds_batch, optim_embeds_batch, after_embeds_batch, target_embeds_batch], dim=1)
            full_input_ids = torch.cat([before_ids_batch, optim_ids_batch, after_ids_batch, target_ids_batch], dim=1)

            attention_mask = torch.ones((current_bs, inputs_embeds.shape[1]), device=self.embedding_device,
                                        dtype=torch.long)

            batch_kwargs = {}
            for k, v in self.vision_kwargs.items():
                if isinstance(v, torch.Tensor):
                    if k == 'pixel_values':
                        batch_kwargs[k] = v.repeat(current_bs, 1)
                    elif k == 'image_grid_thw':
                        batch_kwargs[k] = v.repeat(current_bs, 1)
                    else:
                        try:
                            expand_shape = [current_bs] + list(v.shape[1:])
                            batch_kwargs[k] = v.expand(expand_shape)
                        except RuntimeError:
                            repeat_shape = [current_bs] + [1] * (v.ndim - 1)
                            batch_kwargs[k] = v.repeat(*repeat_shape)
                else:
                    batch_kwargs[k] = v

            outputs = self.model(
                input_ids=full_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                **batch_kwargs
            )

            shift = inputs_embeds.shape[1] - self.target_ids.shape[1]
            shift_logits = outputs.logits[..., shift - 1: -1, :].contiguous()
            shift_labels = self.target_ids.repeat(current_bs, 1).view(-1)

            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels,
                                                     reduction='none')
            batch_loss = loss.view(current_bs, -1).mean(dim=1)
            all_loss.append(batch_loss)

        return torch.cat(all_loss, dim=0)

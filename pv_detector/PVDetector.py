# -*-coding:utf-8-*-
import logging
import random

import numpy as np
import torch
from tqdm import tqdm

from .utils.activations import compute_layer_pv_projection
from .utils.model import get_hidden_states, get_hidden_states_vl, is_vision_language_model
from .utils.other_utils import load_pv_hidden_vector


MODALITIES = ("auto", "text", "vl")


def resolve_detector_modality(model_name, requested_modality):
    if requested_modality not in MODALITIES:
        raise ValueError(f"Unsupported modality: {requested_modality}")

    inferred_modality = "vl" if is_vision_language_model(model_name) else "text"
    if requested_modality == "auto":
        return inferred_modality
    if requested_modality != inferred_modality:
        raise ValueError(
            f"Model {model_name!r} is recognized as {inferred_modality}, "
            f"but modality={requested_modality!r} was requested"
        )
    return requested_modality


def select_layer_values(layer_values, key_layers=None, start_layer=None, end_layer=None):
    if key_layers is not None:
        return [layer_values[layer_idx] for layer_idx in key_layers]
    if start_layer is None or end_layer is None:
        raise ValueError("Either key_layers or both start_layer/end_layer must be provided")
    return layer_values[start_layer:end_layer + 1]


def aggregate_layer_values(layer_values, key_layers=None):
    # Integrate over layer depth, not ROC space; use the raw score for one layer.
    if len(layer_values) == 1:
        return layer_values[0]
    if key_layers is not None:
        return np.trapz(np.array(layer_values), x=np.array(key_layers))
    return np.trapz(np.array(layer_values))


class PVDetector:

    def __init__(
            self,
            target_model,
            tokenizer_or_processor,
            model_name,
            pv_hidden_vector_dir=None,
            modality="auto",
            pv_layers_hidden_vector=None):
        self.target_model = target_model
        self.tokenizer_or_processor = tokenizer_or_processor
        self.model_name = model_name
        self.modality = resolve_detector_modality(model_name, modality)
        if pv_layers_hidden_vector is None:
            if pv_hidden_vector_dir is None:
                raise ValueError(
                    "Either pv_layers_hidden_vector or pv_hidden_vector_dir must be provided"
                )
            pv_layers_hidden_vector = load_pv_hidden_vector(
                data_dir=pv_hidden_vector_dir,
            )
        self.pv_layers_hidden_vector = pv_layers_hidden_vector
        self.threshold = None

    def get_hidden_states(self, input_data, system_prompt):
        # Text and VL inputs share detection logic but use different model inputs.
        if isinstance(input_data, str):
            if self.modality != "text":
                raise TypeError("VL input_data must contain both 'msg' and 'img_path'")
            input_data = {"msg": input_data}
        if not isinstance(input_data, dict):
            raise TypeError("input_data must be a string for text or a dictionary for VL")
        if "msg" not in input_data:
            raise ValueError("input_data is missing required field: msg")

        if self.modality == "text":
            return get_hidden_states(
                self.target_model,
                self.tokenizer_or_processor,
                input_data["msg"],
                system_prompt,
            )
        if "img_path" not in input_data:
            raise ValueError("VL input_data is missing required field: img_path")
        return get_hidden_states_vl(
            self.target_model,
            self.tokenizer_or_processor,
            input_data["msg"],
            input_data["img_path"],
            system_prompt,
            self.model_name,
        )

    def compute_pv_strength(self, input_data, system_prompt, projection_type="proj"):
        hidden_states = self.get_hidden_states(input_data, system_prompt)
        # Score each layer at the final input token of the full chat template.
        last_tok_hidden_states = [layer_hidden[:, -1, :] for layer_hidden in hidden_states]
        last_tok_hidden_states = torch.stack(last_tok_hidden_states, dim=0)
        return compute_layer_pv_projection(
            last_tok_hidden_states,
            self.pv_layers_hidden_vector,
            projection_type=projection_type,
        )

    def detect(
            self, input_data, system_prompt, start_layer=None, end_layer=None, 
            projection_type="proj", key_layers=None):
        if self.threshold is None:
            raise ValueError("Detector threshold must be set before calling detect()")
        label = input_data["label"]
        pv_strength_values = self.compute_pv_strength(
            input_data,
            system_prompt,
            projection_type=projection_type,
        )
        pv_strength_values = select_layer_values(
            pv_strength_values,
            key_layers=key_layers,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        aware_auc = aggregate_layer_values(pv_strength_values, key_layers=key_layers)
        # Larger scores indicate stronger PV alignment; flag scores above the threshold.
        detect_result = 1 if aware_auc > self.threshold else 0
        return label, detect_result

    def detect_batch(
            self, input_datas, system_prompt, start_layer=None, end_layer=None,
            projection_type="proj", key_layers=None):
        true_labels = []
        detect_results = []
        for input_data in tqdm(input_datas, total=len(input_datas), desc="Detecting prompts..."):
            label, detect_result = self.detect(
                input_data,
                system_prompt,
                start_layer=start_layer,
                end_layer=end_layer,
                projection_type=projection_type,
                key_layers=key_layers,
            )
            detect_results.append(detect_result)
            true_labels.append(label)
        return true_labels, detect_results

    def estimate_threshold_from_normal(self, normal_samples, system_prompt,
                start_layer=None, end_layer=None,
                projection_type="proj",
                fpr_target=0.01, n_samples=100, key_layers=None):
        """Estimate thresholds from benign samples (FPR=1% -> 99% quantile)."""
        if len(normal_samples) > n_samples:
            selected = random.sample(normal_samples, n_samples)
        else:
            selected = normal_samples

        scores = []
        for sample in tqdm(selected, desc="Estimating threshold from normal samples"):
            pv_strength = self.compute_pv_strength(
                sample,
                system_prompt,
                projection_type=projection_type,
            )
            pv_strength_values = select_layer_values(
                pv_strength,
                key_layers=key_layers,
                start_layer=start_layer,
                end_layer=end_layer,
            )
            aware_auc = aggregate_layer_values(pv_strength_values, key_layers=key_layers)
            scores.append(aware_auc)
        
        # Calibrate on benign scores: a target FPR of 1% uses the 99th percentile.
        quantile = 1 - fpr_target
        
        
        threshold = np.quantile(scores, quantile)
        logging.info(f"[Threshold] Estimated at FPR={fpr_target:.2%}: threshold = {threshold:.4f}")
        return threshold

    def compute_pv_score_batch(
            self, input_datas, system_prompt, start_layer=None, end_layer=None,
            projection_type="proj", key_layers=None):
        pv_scores = []
        true_labels = []
        for input_data in tqdm(input_datas, total=len(input_datas), desc="Computing pv score..."):
            label = input_data["label"]
            pv_strength_values = self.compute_pv_strength(
                input_data, system_prompt, projection_type=projection_type)
            pv_strength_values = select_layer_values(
                pv_strength_values,
                key_layers=key_layers,
                start_layer=start_layer,
                end_layer=end_layer,
            )
            aware_auc = aggregate_layer_values(pv_strength_values, key_layers=key_layers)
            pv_scores.append(aware_auc)
            true_labels.append(label)
        return true_labels, pv_scores


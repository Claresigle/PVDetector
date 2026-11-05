# -*-coding:utf-8-*-
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch


def get_pv_vector_dir(pv_vector_root, model_name, role_name, sample_num):
    """Return the canonical directory for one set of extracted PV vectors."""
    return os.path.join(pv_vector_root, model_name, role_name, f"{sample_num}_shot")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logging(log_dir, stdout=False, add_time=True):
    if add_time:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f'{log_dir}_{timestamp}.log'
    else:
        log_file_path = f'{log_dir}.log'
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    h_file = logging.FileHandler(log_file_path, mode='w')
    h_stdout = logging.StreamHandler(sys.stdout)
    if stdout:
        logging_cfg = {'level': logging.INFO, 'format': '%(message)s', 'handlers': [h_file, h_stdout]}
    else:
        logging_cfg = {'level': logging.INFO, 'format': '%(message)s', 'handlers': [h_file]}

    logging.basicConfig(**logging_cfg, force=True)


def load_pv_hidden_vector(data_dir):
    """Load the mean-difference PV vectors used by the paper."""
    data_path = os.path.join(data_dir, "layers_pv_vector_mean.pt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found")
    return torch.load(data_path, weights_only=False)


def create_pairs(benign_samples, pv_samples):
    
    # Balance the sets and form random pairs; semantic one-to-one matching is not required.
    if len(benign_samples) < len(pv_samples):
        pv_samples = random.sample(pv_samples, len(benign_samples))
    elif len(benign_samples) > len(pv_samples):
        benign_samples = random.sample(benign_samples, len(pv_samples))
    
    random.shuffle(pv_samples)
    text_pairs = list(zip(benign_samples, pv_samples))
    return text_pairs


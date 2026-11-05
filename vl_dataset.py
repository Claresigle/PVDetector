"""Read VLM samples while preserving each prompt's image mapping."""

import json
import random
from pathlib import Path


def _resolve_image_path(image_path, data_path):
    path = Path(image_path)
    if not path.is_absolute() and not path.is_file():
        path = Path(data_path).parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Image referenced by {data_path} not found: {path}")
    return str(path)


def load_vl_samples(data_path, img_dir=None, sample_num=None):
    """Read records with image paths, or strings paired with an image directory.

    An explicit img_dir overrides record image directories while retaining each
    recorded filename.
    """
    with open(data_path, "r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON list: {data_path}")

    image_dir = Path(img_dir) if img_dir is not None else None
    if image_dir is not None and not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory configured for {data_path} not found: {image_dir}. "
            "Prepare the corresponding images before evaluation."
        )

    samples = []
    for index, record in enumerate(records):
        if isinstance(record, dict):
            message = record.get("msg", record.get("Question"))
            image_path = record.get("img_path")
            if not isinstance(message, str):
                raise ValueError(f"Sample {index} in {data_path} requires a string 'msg' or 'Question'")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"Sample {index} in {data_path} requires 'img_path'")
            if image_dir is not None:
                # Retain the original filename instead of renumbering filtered samples.
                filename = Path(image_path.replace("\\", "/")).name
                image_path = str(image_dir / filename)
        elif isinstance(record, str):
            # GCG string outputs use images copied in output order as img_1.jpg, etc.
            message = record
            if image_dir is None:
                raise ValueError(
                    f"String samples in {data_path} require 'img_dir'; "
                    "otherwise use records containing 'msg' and 'img_path'."
                )
            image_path = str(image_dir / f"img_{index + 1}.jpg")
        else:
            raise TypeError(f"Sample {index} in {data_path} must be a string or an object")
        if image_dir is not None:
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"Configured image for sample {index} not found: {image_path}")
        else:
            image_path = _resolve_image_path(image_path, data_path)
        samples.append((message, image_path))

    if sample_num is None or sample_num == -1:
        return samples
    if not isinstance(sample_num, int) or sample_num < 0:
        raise ValueError("sample_num must be a non-negative integer, -1, or None")
    if sample_num > len(samples):
        raise ValueError(f"Not enough samples in {data_path}: requested={sample_num}, available={len(samples)}")
    # Sample whole text-image pairs to preserve their correspondence.
    return random.sample(samples, sample_num)

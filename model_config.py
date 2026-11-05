"""Shared model defaults and dataset/artifact keys for all entrypoints."""

import posixpath


DEFAULT_TEXT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PHI_VL_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

MODEL_ARGUMENT_HELP = (
    "Hugging Face model ID or local checkpoint directory (default: %(default)s). "
    "For a local model, use --model /path/to/<model-name>."
)


def get_model_key(model_name_or_path):
    """Resolve a Hub ID or local path to the repository's dataset/artifact key."""
    # Use the final path component to look up configs, samples, and vector directories.
    model_name = posixpath.basename(
        posixpath.normpath(str(model_name_or_path).replace("\\", "/"))
    )
    # Existing samples, vector directories, and evaluation configs use this key.
    if model_name == "Llama-3.1-8B-Instruct":
        return "Meta-Llama-3.1-8B-Instruct"
    return model_name

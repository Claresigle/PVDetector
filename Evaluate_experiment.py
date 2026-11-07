import argparse

from model_config import DEFAULT_TEXT_MODEL_ID, MODEL_ARGUMENT_HELP
from pv_detector.experiment_runner import run_text_experiment


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, True, False")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PVDetector on text datasets.")
    parser.add_argument("--model", type=str, default=DEFAULT_TEXT_MODEL_ID, help=MODEL_ARGUMENT_HELP)
    parser.add_argument("--domain_dir", type=str, default="./datasets/Agents_system_prompt")
    parser.add_argument("--custom_dataset_config_path", type=str,
                        default="./configs/dataset_configs/{role}_datasets_config.json")
    parser.add_argument("--dataset_name", default="Alpaca", type=str)
    parser.add_argument("--role_name", type=str, default="RecipeMaster")

    parser.add_argument("--pv_vector_dir", default="./pv_detector/PV_concept_extract/pv_vectors", type=str)
    parser.add_argument("--pv_concept_source_role", default="RecipeMaster", type=str)
    parser.add_argument("--pv_concept_source_model", default="Meta-Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--pv_source_sample_num", type=int, default=30)
    parser.add_argument("--locate_key_layers", type=str_to_bool, default=False)
    parser.add_argument(
        "--locate_strategy",
        default="pv_concept",
        type=str,
        choices=("pos_neg_cos", "pv_concept"),
    )

    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--log_dir", type=str, default="./logs/evaluate_result/")
    parser.add_argument("--output_dir", type=str, default="./result/evaluate_result/")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--estimate_threshold", type=str_to_bool, default=True,
        help="Calibrate from benign samples; False uses a saved preset (default: %(default)s).",
    )
    parser.add_argument(
        "--fpr_target",
        type=float,
        default=0.01,
        help="Target false-positive rate for threshold calibration (default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Delegate initialization, threshold calibration, and evaluation to the shared runner.
    run_text_experiment(args)

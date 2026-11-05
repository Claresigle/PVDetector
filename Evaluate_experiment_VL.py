import argparse

from model_config import DEFAULT_VL_MODEL_ID, MODEL_ARGUMENT_HELP
from pv_detector.experiment_runner import run_vl_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PVDetector on vision-language settings.")
    parser.add_argument("--model", type=str, default=DEFAULT_VL_MODEL_ID, help=MODEL_ARGUMENT_HELP)
    parser.add_argument("--domain_dir", type=str, default="./datasets/Agents_system_prompt")
    parser.add_argument("--custom_dataset_config_path", type=str,
                        default="./configs/dataset_configs/{role}_VL_datasets_config.json")
    parser.add_argument("--dataset_name", default="Alpaca", type=str)
    parser.add_argument("--role_name", type=str, default="RecipeMaster")

    parser.add_argument("--pv_vector_dir", default="./pv_detector/PV_concept_extract/pv_vectors", type=str)
    parser.add_argument("--pv_concept_source_role", default="RecipeMaster", type=str)
    parser.add_argument("--pv_concept_source_model", default="Qwen2.5-VL-7B-Instruct", type=str)
    parser.add_argument("--pv_source_sample_num", type=int, default=30)

    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--log_dir", type=str, default="./logs/evaluate_result/")
    parser.add_argument("--output_dir", type=str, default="./result/evaluate_result/")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--fpr_target", type=float, default=0.01,
        help="Target false-positive rate for threshold calibration (default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # The shared runner loads text-image pairs and invokes the unified detector.
    run_vl_experiment(args)

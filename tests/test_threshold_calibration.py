import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import Evaluate_experiment as text_entry
import Evaluate_experiment_VL as vl_entry
from model_config import get_model_key
from pv_detector import experiment_runner as runner
from pv_detector import locate_pv_aware_layers as locator
from pv_detector.PVDetector import PVDetector


class ThresholdCalibrationTests(unittest.TestCase):
    def parse_args(self, entry, arguments=()):
        with patch.object(sys, "argv", [entry.__name__, *arguments]):
            return entry.parse_args()

    def test_evaluation_entries_default_to_one_percent_and_accept_one_override(self):
        for entry in (text_entry, vl_entry):
            with self.subTest(entry=entry.__name__):
                args = self.parse_args(entry)
                self.assertEqual(args.fpr_target, 0.01)
                self.assertFalse(hasattr(args, "estimate_threshold"))
                self.assertFalse(hasattr(args, "fpr_targets"))
                self.assertEqual(self.parse_args(entry, ["--fpr_target", "0.02"]).fpr_target, 0.02)

    def test_old_switch_and_multiple_targets_are_rejected(self):
        for entry in (text_entry, vl_entry):
            for arguments in (["--estimate_threshold", "False"],
                              ["--fpr_targets", "0.01", "0.02"],
                              ["--fpr_target", "0.01", "0.02"]):
                with self.subTest(entry=entry.__name__, arguments=arguments):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            self.parse_args(entry, arguments)
                    self.assertEqual(error.exception.code, 2)

    def test_detector_default_calibrates_at_the_99th_percentile(self):
        detector = object.__new__(PVDetector)
        detector.compute_pv_strength = Mock(side_effect=lambda sample, *args, **kwargs: [sample["score"]])
        samples = [{"score": float(index)} for index in range(100)]
        threshold = detector.estimate_threshold_from_normal(samples, "policy", key_layers=[0])
        self.assertAlmostEqual(threshold, 98.01)
        self.assertEqual(detector.compute_pv_strength.call_count, 100)

    def exercise_pipeline(self, modality, target=None, dynamic=False, repetitions=1):
        entry = text_entry if modality == "text" else vl_entry
        arguments = ["--fpr_target", str(target)] if target is not None else []
        if dynamic:
            arguments.extend(["--locate_key_layers", "True", "--locate_strategy", "pos_neg_cos"])
        args = self.parse_args(entry, arguments)
        args.pv_concept_source_role = "TripPlanner"
        args.pv_source_sample_num = 12
        model_name = get_model_key(args.model)
        resources = runner.ExperimentResources(
            model_name, object(), object(), {args.role_name: "policy"}, object(),
        )
        detector = Mock()
        detector.estimate_threshold_from_normal.side_effect = [0.25, 0.75][:repetitions]
        detector.detect_batch.side_effect = lambda samples, *a, **kw: (
            [sample["label"] for sample in samples], [sample["label"] for sample in samples],
        )
        config = {model_name: {
            args.dataset_name: {}, "Normal": {"data_path": "unused", "sample_num": -1},
        }}
        loader_name = "load_datasets_samples" if modality == "text" else "load_VL_datasets_samples"
        attack = {"msg": "attack", "label": 1}
        normal = {"msg": "normal-0", "label": 0}
        if modality == "vl":
            attack["img_path"] = normal["img_path"] = "image.jpg"

        def load_samples(groups):
            return {"Normal": [normal]} if "Normal" in groups else {"Alpaca-Ignore_attack": [attack]}

        with patch.object(runner, "set_seed"), patch.object(runner, "set_logging"), \
             patch.object(runner, "load_experiment_resources", return_value=resources), \
             patch.object(runner, "make_detector", return_value=detector), \
             patch.object(runner, "load_config", return_value=config), \
             patch.object(runner, loader_name, side_effect=load_samples), \
             patch.object(runner.os, "makedirs"), \
             patch.object(runner.torch.cuda, "empty_cache"), \
             patch.object(runner, "save_results_csv") as save_csv, \
             patch.object(locator, "locate_pv_aware_key_layers", return_value=[1, 3]) as locate:
            for index in range(repetitions):
                normal = {**normal, "msg": f"normal-{index}"}
                args.pv_source_sample_num = 12 + index
                context = (runner.run_text_experiment if modality == "text" else runner.run_vl_experiment)(args)
                self.assertEqual(detector.threshold, [0.25, 0.75][index])
                self.assertEqual(detector.estimate_threshold_from_normal.call_count, index + 1)
                calibration = detector.estimate_threshold_from_normal.call_args
                self.assertEqual(calibration.args[0], [normal])
                self.assertEqual(calibration.kwargs["fpr_target"], args.fpr_target)
                self.assertEqual(calibration.kwargs["n_samples"], 100)
                self.assertEqual(calibration.kwargs["key_layers"], context.key_layers)
                self.assertEqual(context.key_layers, [1, 3] if dynamic else runner.get_key_layers(
                    model_name, multimodal=(modality == "vl"),
                ))
                self.assertEqual(context.experiment_name, "text_evaluation" if modality == "text" else "vl_evaluation")

        self.assertEqual(detector.detect_batch.call_count, repetitions * 2)
        self.assertEqual(save_csv.call_count, repetitions)
        self.assertEqual(save_csv.call_args.args[0], {
            "Alpaca-Ignore_attack": (-1, 0.0), "Normal": (0.0, -1),
        })
        self.assertTrue(Path(save_csv.call_args.kwargs["output_path"]).name.startswith("fpr_fnr_result_"))
        if dynamic:
            self.assertEqual(locate.call_count, repetitions)
            self.assertEqual(locate.call_args.args[0].locate_strategy, "pos_neg_cos")
        else:
            locate.assert_not_called()

    def test_text_and_vl_calibrate_once_at_the_default_or_requested_fpr(self):
        for modality in ("text", "vl"):
            for target in (None, 0.02):
                with self.subTest(modality=modality, target=target):
                    self.exercise_pipeline(modality, target=target)

    def test_repeated_evaluations_recalibrate_with_the_current_normal_samples(self):
        for modality in ("text", "vl"):
            with self.subTest(modality=modality):
                self.exercise_pipeline(modality, repetitions=2)

    def test_dynamic_text_layers_use_the_same_single_calibration_flow(self):
        self.exercise_pipeline("text", dynamic=True)


if __name__ == "__main__":
    unittest.main()

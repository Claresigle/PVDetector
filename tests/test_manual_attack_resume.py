import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Attack.utils import manual_attack_runner as manual


class ManualAttackResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pvdetector-manual-resume-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_run(self, modality, name, duplicate=False):
        folder = self.root / name
        folder.mkdir()
        messages = ["first", "second", "third"]
        if duplicate:
            messages[1] = messages[0]
        samples = messages
        if modality == "vl":
            samples = [(msg, str(folder / f"img_{index}.jpg"))
                       for index, msg in enumerate(messages)]
            if duplicate:
                samples[1] = samples[0]
        data = manual.ManualAttackData(
            samples=samples,
            save_path=str(folder / "all.json"),
            save_path_no_deny=str(folder / "success.json"),
        )
        context = SimpleNamespace(
            args=SimpleNamespace(attack_strategy="Ignore", dataset="Alpaca", device="cpu"),
            model=object(), tokenizer=object(), processor=object(), model_name="test-model",
            system_prompt="policy", refusal_prefixes=["I cannot"],
            log_path_no_deny=str(folder / "success.log"),
        )
        results = []
        for sample in samples:
            msg = sample[0] if modality == "vl" else sample
            injected = manual.create_attack_msg(msg, "Ignore")
            results.append({"msg": injected, "img_path": sample[1]}
                           if modality == "vl" else injected)
        run = manual.run_vl_manual_attack if modality == "vl" else manual.run_text_manual_attack
        generate = "generate_vl_response" if modality == "vl" else "llm_generate"
        return context, data, results, run, generate

    def read_json(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_resume_after_success_write_before_all_results_commit(self):
        for modality in ("text", "vl"):
            for interrupted_count in (1, 2, 3):
                with self.subTest(modality=modality, interrupted_count=interrupted_count):
                    context, data, results, run, generate = self.make_run(
                        modality, f"{modality}-{interrupted_count}",
                    )
                    original_save = manual.save_json_file

                    def interrupt_commit(path, records):
                        if path == data.save_path and len(records) == interrupted_count:
                            raise RuntimeError("interrupted before commit")
                        original_save(path, records)

                    with patch.object(manual, "_clear_runtime_cache"), \
                         patch.object(manual, generate, return_value="Sure"), \
                         patch.object(manual, "save_json_file", side_effect=interrupt_commit):
                        with self.assertRaisesRegex(RuntimeError, "interrupted before commit"):
                            run(context, data)

                    committed_count = interrupted_count - 1
                    committed = self.read_json(data.save_path) if Path(data.save_path).exists() else []
                    self.assertEqual(committed, results[:committed_count])
                    self.assertEqual(self.read_json(data.save_path_no_deny), results[:interrupted_count])
                    remaining = len(results) - committed_count
                    responses = ["I cannot comply"] + ["Sure"] * (remaining - 1)
                    with patch.object(manual, "_clear_runtime_cache"), \
                         patch.object(manual, generate, side_effect=responses) as resumed:
                        run(context, data)

                    self.assertEqual(resumed.call_count, remaining)
                    self.assertEqual(self.read_json(data.save_path), results)
                    self.assertEqual(self.read_json(data.save_path_no_deny),
                                     results[:committed_count] + results[interrupted_count:])

    def test_legacy_success_only_checkpoint_restarts_from_first_sample(self):
        for modality in ("text", "vl"):
            with self.subTest(modality=modality):
                context, data, results, run, generate = self.make_run(modality, modality)
                manual.save_json_file(data.save_path_no_deny, results[:1])
                with patch.object(manual, "_clear_runtime_cache"), \
                     patch.object(manual, generate, side_effect=["I cannot comply", "Sure", "Sure"]) as resumed:
                    run(context, data)
                self.assertEqual(resumed.call_count, 3)
                self.assertEqual(self.read_json(data.save_path), results)
                self.assertEqual(self.read_json(data.save_path_no_deny), results[1:])

    def test_duplicate_pending_success_does_not_inflate_committed_successes(self):
        for modality in ("text", "vl"):
            with self.subTest(modality=modality):
                context, data, results, run, generate = self.make_run(modality, modality, duplicate=True)
                manual.save_json_file(data.save_path, results[:1])
                manual.save_json_file(data.save_path_no_deny, results[:2])
                with patch.object(manual, "_clear_runtime_cache"), \
                     patch.object(manual, generate, side_effect=["I cannot comply", "Sure"]) as resumed:
                    run(context, data)
                self.assertEqual(resumed.call_count, 2)
                self.assertEqual(self.read_json(data.save_path), results)
                self.assertEqual(self.read_json(data.save_path_no_deny), [results[0], results[2]])

    def test_recovered_checkpoint_is_saved_before_generation_can_fail_again(self):
        for modality in ("text", "vl"):
            with self.subTest(modality=modality):
                context, data, results, run, generate = self.make_run(modality, modality)
                manual.save_json_file(data.save_path, results[:1])
                manual.save_json_file(data.save_path_no_deny, results[:2])
                with patch.object(manual, "_clear_runtime_cache"), \
                     patch.object(manual, generate, side_effect=RuntimeError("generation interrupted")):
                    with self.assertRaisesRegex(RuntimeError, "generation interrupted"):
                        run(context, data)
                self.assertEqual(self.read_json(data.save_path), results[:1])
                self.assertEqual(self.read_json(data.save_path_no_deny), results[:1])

    def test_mismatched_committed_inputs_still_fail_without_overwriting_files(self):
        for modality in ("text", "vl"):
            for mismatch in ("input", "strategy"):
                with self.subTest(modality=modality, mismatch=mismatch):
                    context, data, results, run, generate = self.make_run(modality, f"{modality}-{mismatch}")
                    manual.save_json_file(data.save_path, results[:1])
                    manual.save_json_file(data.save_path_no_deny, results[:2])
                    if mismatch == "strategy":
                        context.args.attack_strategy = "Fake_Complete"
                    else:
                        data.samples[0] = ("changed", data.samples[0][1]) if modality == "vl" else "changed"
                    with patch.object(manual, generate) as resumed:
                        with self.assertRaisesRegex(ValueError, "current inputs, strategy, or image paths"):
                            run(context, data)
                    resumed.assert_not_called()
                    self.assertEqual(self.read_json(data.save_path), results[:1])
                    self.assertEqual(self.read_json(data.save_path_no_deny), results[:2])


if __name__ == "__main__":
    unittest.main()

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PipelineConfigurationTests(unittest.TestCase):
    def make_repository(self, directory):
        root = Path(directory)
        (root / "scripts").mkdir()
        (root / "vladfuzz_runtime").mkdir()
        for name in (
            "load_env.sh", "huggingface_env.sh", "run_vladfuzz.sh",
            "run_pipeline.sh", "localfuzzer_smoke.sh",
        ):
            shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
        shutil.copyfile(
            ROOT / "vladfuzz_runtime/project_env.py",
            root / "vladfuzz_runtime/project_env.py",
        )
        python = root / "python-capture"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1:2] == ['-c'] or (len(sys.argv) > 1 and Path(sys.argv[1]).name == 'project_env.py'):\n"
            "    os.execv(sys.executable, [sys.executable] + sys.argv[1:])\n"
            "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        return root, python

    def environment(self, root, python):
        env = dict(os.environ)
        for key in (
            "MODEL", "MANIFEST", "MANIFEST_ID", "SELECTED_ROOT", "STATIC_ROOT",
            "COMMAND_FILE", "OUTPUT_ROOT", "TEST_CASES_ROOT", "SCENARIO_ID",
            "NUM_COMMANDS", "LLM_PROVIDER", "LLM_MODEL", "VLADFUZZ_MODEL_HOME",
            "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
        ):
            env.pop(key, None)
        env.update(PYTHON_BIN=str(python), CAPTURE=str(root / "arguments.json"))
        return env

    def test_custom_manifest_and_instruction_reach_fuzzer(self):
        with tempfile.TemporaryDirectory() as directory:
            root, python = self.make_repository(directory)
            seed = root / "new_inputs/new_region/seed_1"
            seed.mkdir(parents=True)
            (seed / "command_prior.json").write_text(
                json.dumps({"commands": ["Follow the newly generated route."]})
            )
            manifest = root / "new_inputs/manifest.jsonl"
            manifest.write_text(json.dumps({"id": "new_region/seed_1"}) + "\n")
            env = self.environment(root, python)
            env.update(
                SELECTED_ROOT="new_inputs", MANIFEST="new_inputs/manifest.jsonl",
                STATIC_ROOT="new_static", OUTPUT_ROOT="results/custom/vladfuzz",
            )
            subprocess.run(
                ["bash", "scripts/run_vladfuzz.sh"], cwd=root, env=env,
                check=True, capture_output=True, text=True,
            )
            args = json.loads((root / "arguments.json").read_text())
            for flag, value in (
                ("--manifest", "new_inputs/manifest.jsonl"),
                ("--manifest-id", "new_region/seed_1"),
                ("--instruction", "Follow the newly generated route."),
                ("--output-root", "results/custom/vladfuzz"),
            ):
                self.assertEqual(args[args.index(flag) + 1], value)

    def test_pipeline_keeps_generated_inputs_and_results_separate(self):
        cases = (
            ({}, "test_cases/quickstart", "results/quickstart/vladfuzz"),
            ({"TEST_CASES_ROOT": "new_inputs", "OUTPUT_ROOT": "new_results"},
             "new_inputs", "new_results"),
        )
        for overrides, inputs, results in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                root, python = self.make_repository(directory)
                stages = (
                    "carla_manager.sh", "generate_static_scenarios.sh",
                    "generate_seeds.sh", "generate_instructions.sh",
                    "build_manifest.sh", "run_vladfuzz.sh",
                )
                for name in stages:
                    path = root / "scripts" / name
                    path.write_text(
                        f"#!{sys.executable}\n"
                        "import json, os, sys\n"
                        "from pathlib import Path\n"
                        "keys = ['MODEL', 'SEEDS_ROOT', 'OUTPUT_ROOT', 'SELECTED_ROOT', 'MANIFEST', 'MANIFEST_ID']\n"
                        "row = {'stage': Path(sys.argv[0]).name, 'env': {k: os.getenv(k) for k in keys}}\n"
                        "with open(os.environ['TRACE'], 'a') as f:\n"
                        "    f.write(json.dumps(row) + '\\n')\n",
                        encoding="utf-8",
                    )
                    path.chmod(0o755)
                env = self.environment(root, python)
                env.update(MANAGE_CARLA="0", SCENARIO_ID="new_region",
                           TRACE=str(root / "trace.jsonl"), **overrides)
                subprocess.run(
                    ["bash", "scripts/run_pipeline.sh"], cwd=root, env=env,
                    check=True, capture_output=True, text=True,
                )
                rows = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
                self.assertEqual([row["stage"] for row in rows], list(stages))
                by_stage = {row["stage"]: row["env"] for row in rows}
                self.assertEqual(by_stage["generate_seeds.sh"]["OUTPUT_ROOT"], inputs)
                self.assertEqual(by_stage["generate_instructions.sh"]["SEEDS_ROOT"], inputs)
                fuzz = by_stage["run_vladfuzz.sh"]
                self.assertEqual(fuzz["SELECTED_ROOT"], inputs)
                self.assertEqual(fuzz["MANIFEST"], inputs + "/manifest.jsonl")
                self.assertEqual(fuzz["MANIFEST_ID"], "new_region/seed_1")
                self.assertEqual(fuzz["OUTPUT_ROOT"], results)

    def test_local_smoke_forwards_provider_and_model_from_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            root, python = self.make_repository(directory)
            command = root / "command.json"
            command.write_text(json.dumps({"commands": ["Turn left."]}))
            (root / ".env").write_text(
                "LLM_PROVIDER=gemini\nLLM_MODEL=selected-gemini-model\n"
            )
            env = self.environment(root, python)
            env["COMMAND_FILE"] = str(command)
            subprocess.run(
                ["bash", "scripts/localfuzzer_smoke.sh"], cwd=root, env=env,
                check=True, capture_output=True, text=True,
            )
            args = json.loads((root / "arguments.json").read_text())
            self.assertEqual(args[args.index("--llm-provider") + 1], "gemini")
            self.assertEqual(args[args.index("--llm-model") + 1], "selected-gemini-model")


if __name__ == "__main__":
    unittest.main()

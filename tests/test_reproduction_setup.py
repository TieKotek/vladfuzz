import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from vladfuzz_runtime.model_assets import validate_model_asset
from vladfuzz_runtime.project_env import load_project_env


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_script("test_prepare_models", "scripts/prepare_models.py")
doctor = load_script("test_environment_doctor", "tools/check_environment.py")


class ConfigurationTests(unittest.TestCase):
    def test_launchers_load_settings_before_model_defaults(self):
        for launcher in (ROOT / "scripts").glob("*.sh"):
            source = launcher.read_text()
            if 'MODEL="${MODEL:-' in source:
                self.assertLess(
                    source.index('source "$PROJECT_ROOT/scripts/load_env.sh"'),
                    source.index('MODEL="${MODEL:-'),
                    launcher.name,
                )

    def make_repository(self, directory):
        root = Path(directory)
        (root / "scripts").mkdir()
        (root / "vladfuzz_runtime").mkdir()
        for relative in (
            "scripts/load_env.sh", "scripts/prepare_models.py",
            "vladfuzz_runtime/project_env.py", "vladfuzz_runtime/__init__.py",
            "vladfuzz_runtime/model_assets.py",
        ):
            shutil.copyfile(ROOT / relative, root / relative)
        return root

    def test_shell_and_python_preserve_caller_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            env_file = root / ".env"
            env_file.write_text(
                "CUDA_VISIBLE_DEVICES=4\nVLADFUZZ_TEST_VALUE='path with spaces'\n",
                encoding="utf-8",
            )
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", PYTHON_BIN=sys.executable)
            env.pop("VLADFUZZ_TEST_VALUE", None)
            result = subprocess.run(
                ["bash", "-c", 'source scripts/load_env.sh; '
                 'printf "%s\\n%s" "$CUDA_VISIBLE_DEVICES" "$VLADFUZZ_TEST_VALUE"'],
                cwd=root, env=env, capture_output=True, text=True, check=True,
            )
            self.assertEqual(result.stdout, "2\npath with spaces")
            with patch.dict(os.environ, env, clear=True):
                load_project_env(env_file)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
                self.assertEqual(os.environ["VLADFUZZ_TEST_VALUE"], "path with spaces")

    def test_dotenv_is_not_executed_as_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            marker = root / "must-not-exist"
            (root / ".env").write_text(
                f"VLADFUZZ_TEST_VALUE=$(touch {marker})\n", encoding="utf-8"
            )
            subprocess.run(
                ["bash", "-c", "source scripts/load_env.sh"], cwd=root,
                env=dict(os.environ, PYTHON_BIN=sys.executable), check=True,
            )
            self.assertFalse(marker.exists())

    def test_missing_dotenv_dependency_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("VLADFUZZ_MODEL_HOME=custom\n")
            with patch.dict(sys.modules, {"dotenv": None}):
                with self.assertRaisesRegex(RuntimeError, "python-dotenv"):
                    load_project_env(env_file)

    def test_preparation_reads_dotenv_and_honors_explicit_model_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            (root / "configs").mkdir()
            asset = {
                "id": "test", "backend": "simlingo",
                "method": "huggingface_snapshot", "destination": "base",
                "required_files": ["config.json"],
            }
            (root / "configs/model-assets.json").write_text(
                json.dumps({"assets": [asset]}), encoding="utf-8"
            )
            (root / ".env").write_text("VLADFUZZ_MODEL_HOME=custom\n", encoding="utf-8")
            for folder in ("custom", "explicit"):
                base = root / folder / "base"
                base.mkdir(parents=True)
                (base / "config.json").write_text("{}", encoding="utf-8")
            env = dict(os.environ)
            env.pop("VLADFUZZ_MODEL_HOME", None)
            command = [sys.executable, "scripts/prepare_models.py",
                       "--backend", "simlingo", "--verify-only"]
            for extra, expected in (([], "custom"), (["--model-root", "explicit"], "explicit")):
                completed = subprocess.run(
                    command + extra, cwd=root, env=env, capture_output=True,
                    text=True, check=True,
                )
                self.assertIn(str(root / expected), completed.stdout)


class AssetValidationTests(unittest.TestCase):
    def test_bert_accepts_either_weight_format_but_not_missing_weights(self):
        asset = {
            "method": "huggingface_snapshot", "destination": "bert",
            "required_files": ["config.json"],
            "required_file_alternatives": [["model.safetensors", "pytorch_model.bin"]],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "bert"
            base.mkdir()
            (base / "config.json").write_text("{}")
            self.assertTrue(prepare.verify_asset(asset, root))
            for name in ("model.safetensors", "pytorch_model.bin"):
                (base / name).write_bytes(b"test")
                self.assertEqual(prepare.verify_asset(asset, root), [])
                self.assertEqual(validate_model_asset(asset, root), [])
                (base / name).unlink()

    def test_declared_processor_alias_is_created_without_overwriting(self):
        asset = {
            "method": "huggingface_snapshot", "destination": "base",
            "required_files": ["processor_config.json", "conversation.py"],
            "file_aliases": {"processor_config.json": "preprocessor_config.json"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            (base / "preprocessor_config.json").write_text('{"test": 1}')
            self.assertTrue(prepare.verify_asset(asset, root))
            self.assertFalse((base / "processor_config.json").exists())
            prepare.materialize_file_aliases(asset, root)
            self.assertEqual((base / "processor_config.json").read_text(), '{"test": 1}')
            (base / "conversation.py").write_text("# test")
            self.assertEqual(prepare.verify_asset(asset, root), [])
            (base / "processor_config.json").write_text('{"custom": 2}')
            prepare.materialize_file_aliases(asset, root)
            self.assertEqual((base / "processor_config.json").read_text(), '{"custom": 2}')

    def test_checkpoint_size_and_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.pt").write_bytes(b"test")
            asset = {
                "method": "manual", "destination": "weights.pt", "size": 4,
                "sha256": hashlib.sha256(b"test").hexdigest(),
            }
            self.assertEqual(prepare.verify_asset(asset, root), [])
            asset["sha256"] = "0" * 64
            self.assertTrue(prepare.verify_asset(asset, root))
            asset["size"] = 5
            self.assertTrue(validate_model_asset(asset, root))


class VulkanCheckTests(unittest.TestCase):
    def test_gpu_id_is_checked_after_cuda_visibility_filtering(self):
        cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            get_device_name=lambda index: "NVIDIA test GPU",
        )
        torch = SimpleNamespace(cuda=cuda, __version__="test", version=SimpleNamespace(cuda="test"))
        with patch.dict(sys.modules, {"torch": torch}), \
                patch.object(doctor.shutil, "which", return_value=None), \
                patch.object(doctor, "check_vulkan", return_value=doctor.CheckResult("OK", "Vulkan", "test")):
            self.assertEqual(doctor.check_gpu(0)[-1].level, "OK")
            self.assertEqual(doctor.check_gpu(2)[-1].level, "ERROR")

    def test_headless_launcher_does_not_assume_an_icd_filename(self):
        source = (ROOT / "scripts/carla_manager.sh").read_text()
        self.assertNotIn("/usr/share/vulkan/icd.d/nvidia_icd.json", source)

    @patch.object(doctor.shutil, "which", return_value="/usr/bin/vulkaninfo")
    def test_success_requires_nvidia_device(self, _which):
        for output, expected in (
            ("deviceName = NVIDIA H800", "OK"),
            ("vendorID = 0x10de\ndeviceName = Tesla T4", "OK"),
            ("deviceName = llvmpipe", "ERROR"),
            ("VK_LAYER_NV_optimus NVIDIA Optimus layer\ndeviceName = llvmpipe", "ERROR"),
        ):
            with patch.object(doctor.subprocess, "run", return_value=subprocess.CompletedProcess(
                ["vulkaninfo"], 0, output, ""
            )):
                self.assertEqual(doctor.check_vulkan().level, expected)

    @patch.object(doctor.shutil, "which", return_value="/usr/bin/vulkaninfo")
    def test_broken_driver_and_timeouts_are_errors(self, _which):
        with patch.object(doctor.subprocess, "run", return_value=subprocess.CompletedProcess(
            ["vulkaninfo"], 1, "", "cannot create Vulkan instance"
        )):
            self.assertEqual(doctor.check_vulkan().level, "ERROR")
        with patch.object(doctor.subprocess, "run", side_effect=subprocess.TimeoutExpired(
            ["vulkaninfo"], 30
        )):
            self.assertEqual(doctor.check_vulkan().level, "ERROR")


class ProvenanceTests(unittest.TestCase):
    def test_metadata_records_context_without_credentials(self):
        from vladfuzz_runtime.run_metadata import runtime_provenance, write_metadata

        runtime_provenance.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "DEEPSEEK_API_KEY": "not-for-recording", "CUDA_VISIBLE_DEVICES": "2",
            }):
                original = {"configuration": {"vla_model": "simlingo"}}
                write_metadata(directory, original)
                data = json.loads((Path(directory) / "metadata.json").read_text())
                self.assertNotIn("runtime_provenance", original)
                self.assertNotIn("not-for-recording", json.dumps(data))
                self.assertEqual(data["runtime_provenance"]["environment"]["CUDA_VISIBLE_DEVICES"], "2")
        finally:
            runtime_provenance.cache_clear()


if __name__ == "__main__":
    unittest.main()

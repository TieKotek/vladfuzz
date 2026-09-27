import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelAssetResolutionTests(unittest.TestCase):
    def _complete_bert_dir(self, root: str) -> Path:
        model_dir = Path(root) / "bert-base-uncased"
        model_dir.mkdir()
        for name in ("config.json", "vocab.txt", "model.safetensors"):
            (model_dir / name).write_bytes(b"test")
        return model_dir

    def test_model_root_is_relative_to_repository(self):
        from vladfuzz_runtime.model_assets import resolve_model_root

        resolved = resolve_model_root({"VLADFUZZ_MODEL_HOME": "custom-models"})

        self.assertEqual(resolved, PROJECT_ROOT / "custom-models")

    def test_backend_override_takes_precedence(self):
        from vladfuzz_runtime.model_assets import resolve_bert_model

        with tempfile.TemporaryDirectory() as tmp:
            backend_dir = self._complete_bert_dir(tmp)
            resolved = resolve_bert_model(
                "LMDRIVE_BERT_MODEL",
                env={
                    "LMDRIVE_BERT_MODEL": str(backend_dir),
                    "VLADFUZZ_BERT_MODEL": "/not/used",
                    "VLADFUZZ_STRICT_LOCAL_ASSETS": "1",
                },
            )
        self.assertEqual(resolved, str(backend_dir.resolve()))

    def test_shared_local_model_is_used(self):
        from vladfuzz_runtime.model_assets import resolve_bert_model

        with tempfile.TemporaryDirectory() as tmp:
            shared_dir = self._complete_bert_dir(tmp)
            resolved = resolve_bert_model(
                "LMDRIVE_BERT_MODEL",
                env={
                    "VLADFUZZ_BERT_MODEL": str(shared_dir),
                    "VLADFUZZ_STRICT_LOCAL_ASSETS": "1",
                },
            )
        self.assertEqual(resolved, str(shared_dir.resolve()))

    def test_explicit_incomplete_model_is_rejected(self):
        from vladfuzz_runtime.model_assets import resolve_bert_model

        with tempfile.TemporaryDirectory() as tmp:
            incomplete = Path(tmp) / "bert-base-uncased"
            incomplete.mkdir()
            (incomplete / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                resolve_bert_model(
                    "LMDRIVE_BERT_MODEL",
                    env={"LMDRIVE_BERT_MODEL": str(incomplete)},
                )

    def test_strict_mode_rejects_missing_local_model(self):
        from vladfuzz_runtime.model_assets import resolve_bert_model

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "prepare_models.py"):
                resolve_bert_model(
                    "LMDRIVE_BERT_MODEL",
                    default_local_dir=Path(tmp) / "missing",
                    env={"VLADFUZZ_STRICT_LOCAL_ASSETS": "1"},
                )

    def test_non_strict_mode_uses_official_remote_id(self):
        from vladfuzz_runtime.model_assets import resolve_bert_model

        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_bert_model(
                "LMDRIVE_BERT_MODEL",
                default_local_dir=Path(tmp) / "missing",
                env={},
            )
        self.assertEqual(resolved, "google-bert/bert-base-uncased")


class ModelAssetManifestTests(unittest.TestCase):
    def test_manifest_destinations_are_under_backend_directories(self):
        manifest = json.loads(
            (PROJECT_ROOT / "configs/model-assets.json").read_text(encoding="utf-8")
        )

        for asset in manifest["assets"]:
            destination = Path(asset["destination"])
            self.assertFalse(destination.is_absolute(), asset["id"])
            self.assertNotIn("..", destination.parts, asset["id"])

    def test_task_checkpoints_have_size_and_hash(self):
        manifest = json.loads(
            (PROJECT_ROOT / "configs/model-assets.json").read_text(encoding="utf-8")
        )

        file_assets = [
            asset
            for asset in manifest["assets"]
            if asset["method"] in {"huggingface_file", "manual"}
        ]
        self.assertTrue(file_assets)
        for asset in file_assets:
            self.assertGreater(asset["size"], 0, asset["id"])
            self.assertEqual(len(asset["sha256"]), 64, asset["id"])

    def test_repository_contains_no_model_weight_files(self):
        forbidden_names = (
            "*.bin",
            "*.ckpt",
            "*.gguf",
            "*.onnx",
            "*.safetensors",
            "*.pth",
            "*.pt",
            "*.r50",
        )
        found = []
        for pattern in forbidden_names:
            found.extend(PROJECT_ROOT.glob(f"**/{pattern}"))
        self.assertEqual(found, [])


class ModelAssetIntegrationTests(unittest.TestCase):
    def test_huggingface_helper_uses_project_model_tree(self):
        helper = (PROJECT_ROOT / "scripts/huggingface_env.sh").read_text(encoding="utf-8")
        self.assertNotIn("hf-mirror.com", helper)
        self.assertIn("$VLADFUZZ_MODEL_HOME/shared/bert-base-uncased", helper)
        self.assertNotIn("$PROJECT_ROOT/.cache/models", helper)

    def test_lmdrive_active_blip2_paths_use_shared_resolver(self):
        paths = (
            PROJECT_ROOT / "backends/lmdrive/lmdrive_lavis/models/blip2_models/blip2.py",
            PROJECT_ROOT / "backends/lmdrive/lmdrive_lavis/models/drive_models/blip2.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn("resolve_bert_model", source, str(path))
            self.assertNotIn('from_pretrained("bert-base-uncased"', source, str(path))

    def test_simlingo_source_is_registered_without_embedding_weights(self):
        from vladfuzz_runtime.model_registry import bootstrap_model_environment

        spec = bootstrap_model_environment("simlingo")

        self.assertTrue(spec.source_path.is_file())
        self.assertEqual(spec.config_path, PROJECT_ROOT / "backends/simlingo")

    def test_drivefuzz_disables_expandable_segments_for_old_torch_backends(self):
        source = (PROJECT_ROOT / "scripts/run_drivefuzz.sh").read_text(encoding="utf-8")
        self.assertIn('"lmdrive"|"bevdriver"', source)
        self.assertIn("unset PYTORCH_CUDA_ALLOC_CONF", source)


if __name__ == "__main__":
    unittest.main()

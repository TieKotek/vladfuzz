import os
import signal
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANAGER = PROJECT_ROOT / "scripts" / "carla_manager.sh"


class CarlaManagerScriptTests(unittest.TestCase):
    def test_source_exposes_functions_without_starting_carla(self):
        command = f"source {MANAGER}; type carla_start >/dev/null; type carla_stop >/dev/null"

        result = subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT)

        self.assertEqual(result.returncode, 0)

    def test_start_requires_configured_carla_root(self):
        command = f"source {MANAGER}; unset CARLA_ROOT CARLA_PATH; carla_start test_seed 1"

        result = subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT)

        self.assertNotEqual(result.returncode, 0)

    def test_windowed_mode_does_not_add_offscreen_flags(self):
        command = (
            f"source {MANAGER}; "
            "export VLADFUZZ_CARLA_MODE=windowed; "
            "unset CARLA_LAUNCH_ARGS; "
            "args=\"$(carla_launch_args)\"; "
            "[[ \"$args\" == *-vulkan* && \"$args\" != *-RenderOffScreen* ]]"
        )

        result = subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT)

        self.assertEqual(result.returncode, 0)

    def test_graphics_adapter_applies_without_isolated_mode(self):
        command = (
            f"source {MANAGER}; "
            "export CARLA_GRAPHICS_ADAPTER=1; "
            "export CARLA_ISOLATED=0; "
            "unset CARLA_LAUNCH_ARGS; "
            "args=\"$(carla_launch_args)\"; "
            "[[ \"$args\" == *-graphicsadapter=1* ]]"
        )

        result = subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT)

        self.assertEqual(result.returncode, 0)

    def test_start_and_stop_managed_carla_process(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_root = tmp_path / "carla"
            fake_root.mkdir()
            fake_carla = fake_root / "CarlaUE4.sh"
            fake_carla.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    echo "$@" > launched_args.txt
                    echo "${SDL_VIDEODRIVER:-}" > launched_sdl.txt
                    trap 'exit 0' TERM INT
                    while true; do sleep 1; done
                    """
                ),
                encoding="utf-8",
            )
            fake_carla.chmod(0o755)

            log_dir = tmp_path / "logs"
            command = textwrap.dedent(
                f"""\
                set -euo pipefail
                source {MANAGER}
                export CARLA_ROOT={fake_root}
                export CARLA_LOG_DIR={log_dir}
                export CARLA_SKIP_READY_CHECK=1
                carla_start test_seed 1
                pid="$(cat "$CARLA_LOG_DIR/carla.pid")"
                kill -0 "$pid"
                test -f "$CARLA_LOG_DIR/test_seed_attempt1.log"
                for _ in $(seq 1 20); do
                  if [[ -f {fake_root}/launched_args.txt ]]; then
                    break
                  fi
                  sleep 0.1
                done
                grep -q -- "-RenderOffScreen" {fake_root}/launched_args.txt
                grep -q -- "-nosound" {fake_root}/launched_args.txt
                grep -q -- "-vulkan" {fake_root}/launched_args.txt
                grep -q "dummy" {fake_root}/launched_sdl.txt
                carla_stop
                if kill -0 "$pid" 2>/dev/null; then
                  exit 23
                fi
                """
            )

            result = subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT)

            self.assertEqual(result.returncode, 0)


    def test_default_runtime_logs_are_removed_after_stop(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_root = tmp_path / "carla"
            fake_root.mkdir()
            fake_carla = fake_root / "CarlaUE4.sh"
            fake_carla.write_text(
                "#" + "!/usr/bin/env bash\ntrap exit TERM INT\nwhile true; do sleep 1; done\n",
                encoding="utf-8",
            )
            fake_carla.chmod(0o755)
            run_id = f"carla-manager-test-{os.getpid()}"
            runtime_dir = Path("/tmp/vladfuzz-carla") / run_id
            command = "\n".join(
                [
                    "set -euo pipefail",
                    f"source {MANAGER}",
                    f"export CARLA_ROOT={fake_root}",
                    f"export RUN_ID={run_id}",
                    "export CARLA_SKIP_READY_CHECK=1",
                    "unset CARLA_LOG_DIR KEEP_DEBUG_ARTIFACTS",
                    "carla_start test_seed 1",
                    f"test -f {runtime_dir}/test_seed_attempt1.log",
                    "carla_stop",
                    f"if [[ -e {runtime_dir} ]]; then exit 24; fi",
                ]
            )

            result = subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT)

            self.assertEqual(result.returncode, 0)

    def test_isolated_launch_uses_configured_rpc_port(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_root = tmp_path / "carla"
            fake_root.mkdir()
            fake_carla = fake_root / "CarlaUE4.sh"
            fake_carla.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    echo "$@" > launched_args.txt
                    trap 'exit 0' TERM INT
                    while true; do sleep 1; done
                    """
                ),
                encoding="utf-8",
            )
            fake_carla.chmod(0o755)
            log_dir = tmp_path / "isolated"
            command = textwrap.dedent(
                f"""\
                set -euo pipefail
                source {MANAGER}
                export CARLA_ROOT={fake_root}
                export CARLA_LOG_DIR={log_dir}
                export CARLA_SKIP_READY_CHECK=1
                export CARLA_ISOLATED=1
                export CARLA_PORT=2010
                carla_start test_seed 1
                for _ in $(seq 1 20); do
                  [[ -f {fake_root}/launched_args.txt ]] && break
                  sleep 0.1
                done
                grep -q -- "-carla-rpc-port=2010" {fake_root}/launched_args.txt
                grep -q -- "-carla-streaming-port=2011" {fake_root}/launched_args.txt
                grep -q -- "-carla-secondary-port=2012" {fake_root}/launched_args.txt
                carla_stop
                """
            )

            result = subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT)

            self.assertEqual(result.returncode, 0)

    def test_stopping_isolated_instance_does_not_stop_another_instance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_root = tmp_path / "carla"
            fake_root.mkdir()
            fake_carla = fake_root / "CarlaUE4.sh"
            fake_carla.write_text(
                "#!/usr/bin/env bash\ntrap 'exit 0' TERM INT\nwhile true; do sleep 1; done\n",
                encoding="utf-8",
            )
            fake_carla.chmod(0o755)
            worker_0 = tmp_path / "worker_0"
            worker_1 = tmp_path / "worker_1"
            command = textwrap.dedent(
                f"""\
                set -euo pipefail
                source {MANAGER}
                export CARLA_ROOT={fake_root}
                export CARLA_SKIP_READY_CHECK=1
                export CARLA_ISOLATED=1

                export CARLA_LOG_DIR={worker_0}
                export CARLA_PORT=2000
                carla_start seed_0 1
                pid_0=$(carla_managed_pid)

                export CARLA_LOG_DIR={worker_1}
                export CARLA_PORT=2010
                carla_start seed_1 1
                pid_1=$(carla_managed_pid)

                export CARLA_LOG_DIR={worker_0}
                export CARLA_PORT=2000
                carla_stop
                ! kill -0 "$pid_0" 2>/dev/null
                kill -0 "$pid_1"

                export CARLA_LOG_DIR={worker_1}
                export CARLA_PORT=2010
                carla_stop
                """
            )

            result = subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT)

            self.assertEqual(result.returncode, 0)


    def test_isolated_cleanup_targets_only_matching_rpc_port(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            command = textwrap.dedent(
                f"""\
                set -euo pipefail
                source {MANAGER}
                bash -c 'exec -a /tmp/CarlaUE4-Linux-Shipping python3 -c "import time; time.sleep(300)" CarlaUE4 -carla-rpc-port=2200' &
                pid_0=$!
                bash -c 'exec -a /tmp/CarlaUE4-Linux-Shipping python3 -c "import time; time.sleep(300)" CarlaUE4 -carla-rpc-port=2210' &
                pid_1=$!
                cleanup() {{
                  kill -KILL "$pid_0" "$pid_1" 2>/dev/null || true
                }}
                trap cleanup EXIT
                export CARLA_ISOLATED=1
                export CARLA_PORT=2200
                carla_stop_existing_processes
                if kill -0 "$pid_0" 2>/dev/null; then
                  exit 41
                fi
                kill -0 "$pid_1"
                """
            )

            result = subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT)

            self.assertEqual(result.returncode, 0)



if __name__ == "__main__":
    unittest.main()

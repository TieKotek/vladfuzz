"""Lightweight CUDA memory diagnostics for long-running experiments."""

import json
import os
import time
from collections import deque
from typing import Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in docs/tools envs
    torch = None


class CudaMemoryDiagnostics:
    def __init__(self, log_dir: Optional[str] = None, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = os.environ.get("VLAD_CUDA_DIAGNOSTICS", "1") != "0"
        self.enabled = bool(enabled and torch is not None and torch.cuda.is_available())
        self.log_dir = log_dir or os.environ.get("VLAD_CUDA_DIAG_DIR")
        self.recent = deque(maxlen=int(os.environ.get("VLAD_CUDA_DIAG_RECENT", "80")))
        self.step_interval = max(1, int(os.environ.get("VLAD_CUDA_DIAG_STEP_INTERVAL", "100")))
        self.last_snapshot = None
        if self.enabled and self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

    @property
    def jsonl_path(self):
        if not self.log_dir:
            return None
        return os.path.join(self.log_dir, "cuda_memory.jsonl")

    @property
    def recent_path(self):
        if not self.log_dir:
            return None
        return os.path.join(self.log_dir, "cuda_memory_recent.json")

    @property
    def oom_path(self):
        if not self.log_dir:
            return None
        return os.path.join(self.log_dir, "cuda_oom_context.json")

    def snapshot(self, event: str, **extra) -> Optional[Dict]:
        if not self.enabled:
            return None
        device = torch.cuda.current_device()
        snapshot = {
            "timestamp": time.time(),
            "event": event,
            "device": int(device),
            "allocated_mb": round(torch.cuda.memory_allocated(device) / 1048576, 2),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / 1048576, 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1048576, 2),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1048576, 2),
        }
        snapshot.update(extra)
        self.last_snapshot = snapshot
        self.recent.append(snapshot)
        self._write_jsonl(snapshot)
        return snapshot

    def maybe_step(self, step: int, event: str = "step", **extra) -> Optional[Dict]:
        if step <= 0 or step % self.step_interval != 0:
            return None
        return self.snapshot(event, step=step, **extra)

    def record_oom(self, exc: BaseException, **extra) -> Optional[Dict]:
        if not self.enabled:
            return None
        snapshot = self.snapshot("cuda_oom", exception=repr(exc), **extra)
        self._write_recent()
        return snapshot

    def _write_jsonl(self, snapshot: Dict) -> None:
        path = self.jsonl_path
        if not path:
            return
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    def _write_recent(self) -> None:
        if self.recent_path:
            with open(self.recent_path, "w", encoding="utf-8") as file:
                json.dump(list(self.recent), file, indent=2)
        if self.oom_path:
            with open(self.oom_path, "w", encoding="utf-8") as file:
                json.dump({
                    "last_snapshot": self.last_snapshot,
                    "recent": list(self.recent),
                }, file, indent=2)


def get_cuda_diagnostics(agent=None) -> CudaMemoryDiagnostics:
    diag = getattr(agent, "cuda_diagnostics", None) if agent is not None else None
    if diag is None:
        diag = CudaMemoryDiagnostics()
        if agent is not None:
            agent.cuda_diagnostics = diag
    return diag

import json
import os
from datetime import datetime
from typing import Dict, Optional


class HourlyCheckpointRecorder:
    """Write cumulative experiment counters at fixed wall-clock intervals."""

    def __init__(
        self,
        output_dir: str,
        start_time: float,
        interval_seconds: float = 3600.0,
        method: Optional[str] = None,
        model: Optional[str] = None,
        manifest_id: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.start_time = start_time
        self.interval_seconds = interval_seconds
        self.method = method
        self.model = model
        self.manifest_id = manifest_id
        self.next_hour = 1
        self.checkpoints = []
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def json_path(self) -> str:
        return os.path.join(self.output_dir, "hourly_checkpoints.json")

    @property
    def jsonl_path(self) -> str:
        return os.path.join(self.output_dir, "hourly_checkpoints.jsonl")

    def update(
        self,
        now: float,
        executions: int,
        failures: int,
        extra: Optional[Dict] = None,
    ) -> None:
        elapsed = max(0.0, now - self.start_time)
        while elapsed >= self.next_hour * self.interval_seconds:
            checkpoint = self._make_checkpoint(now, executions, failures, extra or {})
            self.checkpoints.append(checkpoint)
            self._write(checkpoint)
            self.next_hour += 1

    def _make_checkpoint(self, now: float, executions: int, failures: int, extra: Dict) -> Dict:
        checkpoint = {
            "hour": self.next_hour,
            "target_elapsed_seconds": round(self.next_hour * self.interval_seconds, 2),
            "recorded_elapsed_seconds": round(max(0.0, now - self.start_time), 2),
            "timestamp": datetime.fromtimestamp(now).isoformat(),
            "executions": int(executions),
            "failures": int(failures),
            "failure_rate": round(failures / executions, 6) if executions else 0.0,
        }
        if self.method is not None:
            checkpoint["method"] = self.method
        if self.model is not None:
            checkpoint["model"] = self.model
        if self.manifest_id is not None:
            checkpoint["manifest_id"] = self.manifest_id
        checkpoint.update(extra)
        return checkpoint

    def _write(self, checkpoint: Dict) -> None:
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(self.checkpoints, file, indent=2)
        with open(self.jsonl_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(checkpoint, ensure_ascii=False) + "\n")

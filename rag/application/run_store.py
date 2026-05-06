import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path("data/runs/ask_runs")


class RunStore:
    """Persist RAG run results for debugging, tracing, and later observability."""

    def __init__(self, runs_dir: Path = DEFAULT_RUNS_DIR) -> None:
        self.runs_dir = runs_dir

    def write_ask_run(self, result: dict[str, Any]) -> Path:
        """Write one RAG ask result to disk as JSON."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc)
        run_id = result["run_id"]

        output_path = self.runs_dir / f"{run_id}.json"

        payload = {
            "created_at": created_at.isoformat(),
            **result,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        logger.info("Wrote ask run to %s", output_path)

        return output_path

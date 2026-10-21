"""Append-only JSON-lines log mirrored to stdout."""
import json
import time
from pathlib import Path


class JsonlLog:
    """One JSON object per line, timestamped."""

    def __init__(self, path):
        self.path = Path(path)

    def write(self, **row) -> None:
        line = json.dumps({"time": round(time.time(), 1), **row})
        with self.path.open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

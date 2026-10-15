#Local JSON and text cache helpers for repeatable SEC research runs.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative_path: str) -> Path:
        return self.root / relative_path

    def exists(self, relative_path: str) -> bool:
        return self.path(relative_path).exists()

    def write_text(self, relative_path: str, text: str) -> Path:
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def read_text(self, relative_path: str) -> str:
        return self.path(relative_path).read_text(encoding="utf-8")

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def read_json(self, relative_path: str) -> Any:
        return json.loads(self.path(relative_path).read_text(encoding="utf-8"))

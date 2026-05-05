"""I/O utilities for JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read a JSON file and return parsed content."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write data to a JSON file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

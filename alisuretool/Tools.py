"""Small drop-in subset of alisuretool.Tools used by this repro package."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Tools:
    @staticmethod
    def print(message: Any, txt_path: str | None = None, **_: Any) -> None:
        text = str(message)
        print(text, flush=True)
        if txt_path:
            path = Path(txt_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")

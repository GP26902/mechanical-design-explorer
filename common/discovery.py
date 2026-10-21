"""Chapter auto-discovery. Pure module (no Streamlit) so it can be unit tested.

A chapter is any chapters/<name>.py (not starting with '_') that defines TITLE, DESCRIPTION
and a callable render(). It may also define KEY_PREFIX and ABOUT (both optional). Broken chapters are reported, never fatal.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REQUIRED = ("TITLE", "DESCRIPTION", "render")


@dataclass
class Chapter:
    name: str                              # file stem
    title: str
    description: str
    render: Optional[Callable[[], None]]
    error: Optional[str] = None            # set when the chapter could not be loaded
    key_prefix: str = ""                   # optional KEY_PREFIX: session keys the Reset button clears
    about: str = ""                        # optional ABOUT: markdown shown in the sidebar expander

    @property
    def ok(self) -> bool:
        return self.error is None


def discover_chapters(chapters_dir: Path) -> list[Chapter]:
    """Import every chapter file in chapters_dir, sorted by filename."""
    chapters_dir = Path(chapters_dir)
    root = str(chapters_dir.resolve().parent)
    if root not in sys.path:               # chapters import from common/
        sys.path.insert(0, root)

    found: list[Chapter] = []
    for path in sorted(chapters_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        found.append(_load(path))
    return found


def _load(path: Path) -> Chapter:
    name = path.stem
    try:
        spec = importlib.util.spec_from_file_location(f"chapter_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        last = traceback.format_exc().strip().splitlines()[-1]
        return Chapter(name, name, "", None, error=f"Could not import {path.name}: {last}")

    missing = [a for a in REQUIRED if not hasattr(module, a)]
    if missing or not callable(getattr(module, "render", None)):
        problem = ", ".join(missing) if missing else "render (not callable)"
        return Chapter(name, name, "", None, error=f"{path.name} is missing: {problem}")

    return Chapter(name, str(module.TITLE), str(module.DESCRIPTION), module.render, None,
                   str(getattr(module, "KEY_PREFIX", "")), str(getattr(module, "ABOUT", "")))

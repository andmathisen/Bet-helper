from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    # This file lives at <repo>/bet_helper/storage.py
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    p = repo_root() / "data"
    p.mkdir(exist_ok=True)
    return p


def historical_path(league: str) -> Path:
    """
    Canonical location for historical matches.
    We keep historical data under data/ to match upcoming/predictions.
    """
    return data_dir() / f"historical_matches_{league}.json"


def legacy_historical_path(league: str) -> Path:
    """Old location kept for one-time migration."""
    return repo_root() / f"historical_matches_{league}.json"


def migrate_legacy_historical_files() -> list[tuple[str, str]]:
    """
    Move any `historical_matches_*.json` from repo root into data/.
    Returns list of (from,to) pairs for moved files.
    """
    moved: list[tuple[str, str]] = []
    root = repo_root()
    dst_dir = data_dir()
    for p in root.glob("historical_matches_*.json"):
        try:
            dst = dst_dir / p.name
            if dst.exists():
                continue
            # atomic move if possible
            p.rename(dst)
            moved.append((str(p), str(dst)))
        except Exception:
            # fallback: copy then delete
            try:
                dst = dst_dir / p.name
                if dst.exists():
                    continue
                dst.write_bytes(p.read_bytes())
                p.unlink()
                moved.append((str(p), str(dst)))
            except Exception:
                continue
    return moved


def upcoming_path(league: str) -> Path:
    return data_dir() / f"upcoming_{league}.json"


def predictions_path(league: str) -> Path:
    return data_dir() / f"predictions_{league}.json"


def load_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


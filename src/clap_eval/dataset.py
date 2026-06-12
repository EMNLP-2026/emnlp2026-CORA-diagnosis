from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_cora_jsonl(path: Path | str) -> list[dict]:
    """Load the local CORA JSONL file prepared by the user."""
    jsonl = Path(path)
    return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_audio_path(record: dict, audio_dir: Path | str) -> Path:
    """Resolve an audio file for a CORA record from an explicit local audio directory."""
    root = Path(audio_dir)
    metadata = record.get("metadata", {}) or {}
    raw_candidates: Iterable[object] = (
        metadata.get("file_name"),
        metadata.get("relative_path"),
        metadata.get("path"),
        metadata.get("audio_file"),
        metadata.get("source_audio_path"),
        record.get("audio_id"),
        f"{record.get('audio_id')}.wav" if record.get("audio_id") else None,
    )
    dataset = str(record.get("dataset") or record.get("dataset_slug") or "")
    for raw in raw_candidates:
        if not raw:
            continue
        candidate = Path(str(raw))
        paths = [candidate] if candidate.is_absolute() else []
        paths.extend([root / candidate, root / dataset / candidate])
        for path in paths:
            if path.exists():
                return path
    raise FileNotFoundError(f"Could not resolve audio for record {record.get('audio_id')} under {root}")

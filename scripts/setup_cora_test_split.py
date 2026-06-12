#!/usr/bin/env python3
"""Download the CORA Hugging Face test split into the local experiment layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "msnowchanj/CORA"
SPLIT = "test"
QUERY_FORMS = ("key_phrase", "statement", "question", "command", "indirect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the CORA test split from Hugging Face.")
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--name", default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data/cora/test")
    parser.add_argument("--jsonl-output", type=Path, default=None)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--audio-column", default=None, help="Override automatic audio column detection.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_audio_feature(feature: object) -> bool:
    return feature.__class__.__name__ == "Audio" or str(feature).startswith("Audio")


def detect_audio_column(dataset: Any, explicit: str | None) -> str | None:
    if explicit:
        if explicit not in dataset.column_names:
            raise ValueError(f"Audio column not found in dataset: {explicit}")
        return explicit
    for name, feature in getattr(dataset, "features", {}).items():
        if is_audio_feature(feature):
            return name
    for name in ("audio", "wav", "sound", "file"):
        if name in dataset.column_names:
            return name
    return None


def safe_name(value: object, index: int) -> str:
    raw = str(value or f"sample_{index:06d}")
    keep = [ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw]
    name = "".join(keep).strip("._")
    return name or f"sample_{index:06d}"


def jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if not isinstance(v, bytes)}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def audio_extension(audio: Any) -> str:
    path = None
    if isinstance(audio, dict):
        path = audio.get("path")
    elif isinstance(audio, str):
        path = audio
    suffix = Path(str(path)).suffix if path else ""
    return suffix or ".wav"


def write_audio(audio: Any, audio_dir: Path, audio_id: str, index: int) -> str | None:
    if audio is None:
        return None

    ext = audio_extension(audio)
    out_path = audio_dir / f"{safe_name(audio_id, index)}{ext}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, dict):
        payload = audio.get("bytes")
        source_path = audio.get("path")
        array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
        if payload:
            out_path.write_bytes(payload)
            return out_path.name
        if source_path and Path(source_path).exists():
            source = Path(source_path)
            if source.resolve() != out_path.resolve():
                shutil.copyfile(source, out_path)
            return out_path.name
        if array is not None and sampling_rate is not None:
            import soundfile as sf

            sf.write(out_path, array, int(sampling_rate))
            return out_path.name
        return None

    if isinstance(audio, bytes):
        out_path.write_bytes(audio)
        return out_path.name

    if isinstance(audio, str) and Path(audio).exists():
        source = Path(audio)
        if source.suffix:
            out_path = audio_dir / f"{safe_name(audio_id, index)}{source.suffix}"
        if source.resolve() != out_path.resolve():
            shutil.copyfile(source, out_path)
        return out_path.name

    return None


def normalize_record(row: dict[str, Any], audio_column: str | None, audio_rel: str | None, index: int) -> dict[str, Any]:
    record = dict(row)
    if audio_column:
        record.pop(audio_column, None)

    record = jsonable(record)
    audio_id = record.get("audio_id") or record.get("clip_id") or record.get("id") or f"sample_{index:06d}"
    record["audio_id"] = str(audio_id)

    generated = dict(record.get("generated_queries") or {})
    for form in QUERY_FORMS:
        if form not in generated and form in record:
            generated[form] = record[form]
    if generated:
        record["generated_queries"] = generated

    metadata = dict(record.get("metadata") or {})
    if audio_rel:
        metadata["relative_path"] = audio_rel
        metadata.setdefault("file_name", Path(audio_rel).name)
    if metadata:
        record["metadata"] = metadata
    return record


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    jsonl_output = args.jsonl_output or output_root / "eq_by_clip.jsonl"
    audio_dir = args.audio_dir or output_root / "audio"

    if jsonl_output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {jsonl_output}. Pass --overwrite to replace it.")

    try:
        from datasets import Audio, load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: datasets. Run `uv sync` first.") from exc

    load_kwargs: dict[str, Any] = {"split": args.split}
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    dataset = load_dataset(args.dataset, args.name, **load_kwargs) if args.name else load_dataset(args.dataset, **load_kwargs)
    audio_column = detect_audio_column(dataset, args.audio_column)
    if audio_column:
        try:
            dataset = dataset.cast_column(audio_column, Audio(decode=False))
        except Exception:
            pass

    output_root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = 0
    audio_files = 0
    with jsonl_output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            audio_value = row.get(audio_column) if audio_column else None
            audio_id = row.get("audio_id") or row.get("clip_id") or row.get("id") or f"sample_{index:06d}"
            audio_rel = write_audio(audio_value, audio_dir, str(audio_id), index)
            if audio_rel:
                audio_files += 1
            record = normalize_record(dict(row), audio_column, audio_rel, index)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1

    print(f"Downloaded {args.dataset}:{args.split}")
    print(f"Rows: {rows}")
    print(f"JSONL: {jsonl_output}")
    print(f"Audio files: {audio_files} -> {audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

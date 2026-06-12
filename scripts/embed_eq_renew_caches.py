#!/usr/bin/env -S uv run python
"""Build EQ_renew embedding caches for base CLAP, OEA, and RobustCLAP models.

This script is intentionally resumable: each model writes its own cache files,
and completed model caches are skipped unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clap_eval.models import get_model  # noqa: E402

BASE_MODELS = ("laion", "msclap", "m2d", "mga")
EQ_FORMS = ("key_phrase", "statement", "question", "command", "indirect")
SOURCE_FORMS = ("source_caption", "statement", "full_caption")
DEFAULT_EQ_JSONL = PROJECT_ROOT / "data/cora/test/eq_by_clip.jsonl"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data/cora/test/audio"
DEFAULT_BASE_CACHE_DIR = PROJECT_ROOT / "results/eq_renew/vanilla_multiclap_eq_embedding_cache"
DEFAULT_SOURCE_CACHE_DIR = PROJECT_ROOT / "results/eq_renew/single_model_invariance_embedding_cache"
DEFAULT_EXTERNAL_CACHE_ROOT = PROJECT_ROOT / "results/eq_renew/external_retriever_embedding_cache"

EXTERNAL_RUNS = {
    "oea_qwen3b_cl": ("oea", "OEA-Qwen3B-Cl"),
    "oea_qwen3b_ac": ("oea", "OEA-Qwen3B-AC"),
    "robustclap": ("robustclap", None),
}


@dataclass(frozen=True)
class Record:
    audio_id: str
    dataset: str
    dataset_slug: str
    audio_path: Path
    texts: dict[str, str]


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def model_config(config: dict, model_name: str) -> dict:
    for item in config.get("models", []):
        if str(item.get("name", "")).lower() == model_name:
            cfg = dict(item)
            for key in ("repo_path", "checkpoint_path", "model_path"):
                value = cfg.get(key)
                if value:
                    p = Path(value)
                    cfg[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
            return cfg
    raise ValueError(f"Missing model config for {model_name}")


def external_config(config: dict, label: str) -> dict:
    cfg = dict((config.get("external_retrievers") or {}).get(label) or {})
    for key in ("repo_path", "base_model_path", "checkpoint_path"):
        value = cfg.get(key)
        if value:
            p = Path(value)
            cfg[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def resolve_audio_path(row: dict, audio_dir: Path) -> Path | None:
    metadata = row.get("metadata", {}) or {}
    raw_candidates = [
        metadata.get("file_name"),
        metadata.get("relative_path"),
        metadata.get("path"),
        metadata.get("audio_file"),
        metadata.get("source_audio_path"),
        row.get("audio_id"),
        f"{row.get('audio_id', '')}.wav" if row.get("audio_id") else None,
    ]
    dataset = str(row.get("dataset") or row.get("dataset_slug") or "")
    for raw in raw_candidates:
        if not raw:
            continue
        p = Path(str(raw))
        candidates = [p] if p.is_absolute() else []
        candidates.extend([audio_dir / p, audio_dir / dataset / p])
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def load_records(eq_jsonl: Path, audio_dir: Path, limit: int | None) -> list[Record]:
    records: list[Record] = []
    for line in eq_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        generated = row.get("generated_queries", {}) or {}
        texts = {form: str(generated.get(form, "")).strip() for form in EQ_FORMS}
        texts["source_caption"] = str(row.get("source_caption", "")).strip()
        texts["full_caption"] = str(generated.get("full_caption") or row.get("full_caption") or "").strip()
        if any(not texts[form] for form in (*EQ_FORMS, "source_caption")):
            continue
        audio_path = resolve_audio_path(row, audio_dir)
        if audio_path is None:
            continue
        records.append(
            Record(
                audio_id=str(row.get("audio_id", len(records))),
                dataset=str(row.get("dataset") or "unknown"),
                dataset_slug=str(row.get("dataset_slug") or f"{row.get('dataset') or 'unknown'}_test"),
                audio_path=audio_path,
                texts=texts,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"No usable records found in {eq_jsonl}")
    return records


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def compute_audio_embeddings(model, records: list[Record], batch_size: int, desc: str) -> np.ndarray:
    chunks = []
    for batch in tqdm(list(batched(records, batch_size)), desc=desc, unit="batch"):
        loaded = [load_audio(record.audio_path) for record in batch]
        by_sr: dict[int, list[int]] = {}
        for i, (_, sr) in enumerate(loaded):
            by_sr.setdefault(sr, []).append(i)
        batch_embeddings: list[np.ndarray | None] = [None] * len(batch)
        for sr, positions in by_sr.items():
            audio_data = [loaded[i][0] for i in positions]
            embeddings = model.get_audio_embedding(audio_data, sr)
            for pos, emb in zip(positions, embeddings):
                batch_embeddings[pos] = emb
        chunks.append(l2(np.stack(batch_embeddings, axis=0)))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def compute_text_embeddings(model, records: list[Record], forms: tuple[str, ...], batch_size: int, desc: str) -> dict[str, np.ndarray]:
    result = {}
    for form in forms:
        texts = [record.texts[form] for record in records]
        chunks = []
        for batch in tqdm(list(batched(texts, batch_size)), desc=f"{desc} {form}", unit="batch"):
            chunks.append(l2(model.get_text_embedding(batch)))
        result[form] = np.concatenate(chunks, axis=0).astype(np.float32)
    return result


def cache_complete(paths: list[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def build_base_model_cache(args: argparse.Namespace, config: dict, records: list[Record], model_name: str) -> None:
    audio_path = args.base_cache_dir / f"eq_renew_audio_{model_name}_test.npy"
    text_path = args.base_cache_dir / f"eq_renew_text_{model_name}_test.npz"
    source_text_path = args.source_cache_dir / f"eq_renew_source_text_{model_name}_test.npz"
    manifest_path = args.base_cache_dir / f"eq_renew_{model_name}_manifest.json"
    done_paths = [audio_path, text_path, source_text_path, manifest_path]
    if not args.overwrite and cache_complete(done_paths):
        print(f"[skip] {model_name}: cache already complete", flush=True)
        return

    print(f"[base] loading {model_name} on {args.device}", flush=True)
    model = get_model(model_name, model_config(config, model_name), args.device)
    audio = compute_audio_embeddings(model, records, args.base_batch_size, f"[{model_name}] audio")
    atomic_save_npy(audio_path, audio)
    text = compute_text_embeddings(model, records, EQ_FORMS, args.base_batch_size, f"[{model_name}] text")
    atomic_save_npz(text_path, text)
    source_text = compute_text_embeddings(model, records, SOURCE_FORMS, args.base_batch_size, f"[{model_name}] source")
    atomic_save_npz(source_text_path, source_text)
    atomic_write_json(
        manifest_path,
        {
            "dataset": "EQ_renew",
            "split": "test",
            "model": model_name,
            "num_items": len(records),
            "eq_jsonl": str(args.eq_jsonl),
            "audio_dir": str(args.audio_dir),
            "audio_embeddings": str(audio_path),
            "text_embeddings": str(text_path),
            "source_text_embeddings": str(source_text_path),
            "eq_forms": list(EQ_FORMS),
            "source_forms": list(SOURCE_FORMS),
        },
    )
    print(f"[done] {model_name}: wrote {audio_path.parent} and {source_text_path.parent}", flush=True)
    del model
    gc.collect()
    try:
        import torch

        if "cuda" in str(args.device):
            torch.cuda.empty_cache()
    except Exception:
        pass


def external_done(cache_root: Path, label: str) -> bool:
    cache_dir = cache_root / label
    return cache_complete(
        [
            cache_dir / "audio_embeddings.npy",
            cache_dir / "text_embeddings.npz",
            cache_dir / "records.jsonl",
            cache_dir / "metadata.json",
        ]
    )


def run_external_cache(args: argparse.Namespace, config: dict, label: str) -> None:
    family, checkpoint = EXTERNAL_RUNS[label]
    ext_cfg = external_config(config, label)
    if not args.overwrite and external_done(args.external_cache_root, label):
        print(f"[skip] {label}: cache already complete", flush=True)
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_external_retriever_embeddings.py"),
        "--model",
        family,
        "--model-label",
        label,
        "--eq-jsonl",
        str(args.eq_jsonl),
        "--audio-dir",
        str(args.audio_dir),
        "--output-root",
        str(args.external_cache_root),
        "--device",
        args.device,
        "--audio-batch-size",
        str(args.external_audio_batch_size),
        "--text-batch-size",
        str(args.external_text_batch_size),
    ]
    if family == "oea":
        if checkpoint is not None:
            cmd.extend(["--oea-checkpoint", checkpoint])
        if ext_cfg.get("repo_path"):
            cmd.extend(["--oea-repo-path", ext_cfg["repo_path"]])
        if ext_cfg.get("base_model_path"):
            cmd.extend(["--oea-base-model-path", ext_cfg["base_model_path"]])
        if ext_cfg.get("checkpoint_path"):
            cmd.extend(["--oea-checkpoint-path", ext_cfg["checkpoint_path"]])
    elif family == "robustclap":
        if ext_cfg.get("repo_path"):
            cmd.extend(["--robustclap-repo-path", ext_cfg["repo_path"]])
        if ext_cfg.get("checkpoint_path"):
            cmd.extend(["--robustclap-checkpoint", ext_cfg["checkpoint_path"]])
    if args.overwrite:
        # The external generator overwrites files in-place when rerun.
        pass
    print("[external] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print(f"[done] {label}: wrote {args.external_cache_root / label}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EQ_renew embedding caches for CLAP4 + OEA Qwen3B + RobustCLAP.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--eq-jsonl", type=Path, default=DEFAULT_EQ_JSONL)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--base-cache-dir", type=Path, default=DEFAULT_BASE_CACHE_DIR)
    parser.add_argument("--source-cache-dir", type=Path, default=DEFAULT_SOURCE_CACHE_DIR)
    parser.add_argument("--external-cache-root", type=Path, default=DEFAULT_EXTERNAL_CACHE_ROOT)
    parser.add_argument("--device", default=os.environ.get("CORA_EMBED_DEVICE", "cuda:1"))
    parser.add_argument("--base-batch-size", type=int, default=8)
    parser.add_argument("--external-audio-batch-size", type=int, default=4)
    parser.add_argument("--external-text-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="+", default=[*BASE_MODELS, *EXTERNAL_RUNS.keys()])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [item.lower() for item in args.only]
    valid = set(BASE_MODELS) | set(EXTERNAL_RUNS)
    unknown = sorted(set(selected) - valid)
    if unknown:
        raise SystemExit(f"Unknown --only target(s): {', '.join(unknown)}")
    if not args.eq_jsonl.exists():
        raise FileNotFoundError(f"Missing EQ_renew jsonl: {args.eq_jsonl}. Prepare the CORA JSONL/audio files manually and pass --eq-jsonl/--audio-dir.")
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Missing EQ_renew audio dir: {args.audio_dir}. Prepare the CORA JSONL/audio files manually and pass --eq-jsonl/--audio-dir.")

    config = load_config(args.config)
    records = load_records(args.eq_jsonl, args.audio_dir, args.limit)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.dataset] = counts.get(record.dataset, 0) + 1
    print(f"[data] records={len(records)} datasets={counts}", flush=True)

    args.base_cache_dir.mkdir(parents=True, exist_ok=True)
    args.source_cache_dir.mkdir(parents=True, exist_ok=True)
    args.external_cache_root.mkdir(parents=True, exist_ok=True)

    for model_name in BASE_MODELS:
        if model_name in selected:
            if args.dry_run:
                print(f"[dry-run] would build base cache for {model_name}", flush=True)
            else:
                build_base_model_cache(args, config, records, model_name)

    for label in EXTERNAL_RUNS:
        if label in selected:
            run_external_cache(args, config, label)

    print("[all done] EQ_renew embedding cache tasks finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

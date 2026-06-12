#!/usr/bin/env python3
"""Check downloaded CORA data, local model paths, and optional caches."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_config() -> dict:
    return {
        "datasets": {"cora_jsonl": "data/cora/test/eq_by_clip.jsonl", "audio_dir": "data/cora/test/audio"},
        "runtime": {
            "base_cache_dir": "results/eq_renew/vanilla_multiclap_eq_embedding_cache",
            "source_cache_dir": "results/eq_renew/single_model_invariance_embedding_cache",
            "external_cache_root": "results/eq_renew/external_retriever_embedding_cache",
        },
        "models": [
            {"name": "laion", "model_path": "models/laion-clap-htsat-unfused"},
            {"name": "mga", "repo_path": "models_third_party/MGA-CLAP", "checkpoint_path": "models/mga/mga-clap.pt"},
            {"name": "msclap", "checkpoint_path": "models/msclap/msclap_2023.pt"},
            {"name": "m2d", "repo_path": "models_third_party/m2d", "checkpoint_path": "models/m2d/checkpoint-30.pth"},
        ],
        "external_retrievers": {
            "oea_qwen3b_cl": {"repo_path": "models_third_party/Omni-Embed-Audio", "base_model_path": "models/oea/OEA-Qwen3B-Cl/base_model", "checkpoint_path": "models/oea/OEA-Qwen3B-Cl/checkpoint.pt"},
            "oea_qwen3b_ac": {"repo_path": "models_third_party/Omni-Embed-Audio", "base_model_path": "models/oea/OEA-Qwen3B-AC/base_model", "checkpoint_path": "models/oea/OEA-Qwen3B-AC/checkpoint.pt"},
            "robustclap": {"repo_path": "models_third_party/linguistic_robust_clap", "checkpoint_path": "models/robustclap/630k-audioset-best.pt"},
        },
    }


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError:
        print("[warn] PyYAML is not installed; using built-in default paths. Install dependencies or run with uv for custom config parsing.")
        return default_config()
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}




def absolutize(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def add_check(rows: list[tuple[str, Path | None, bool, str]], name: str, path: Path | None, required: bool = True) -> None:
    ok = bool(path and path.exists())
    if path is None and not required:
        ok = True
    rows.append((name, path, ok, "required" if required else "optional"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate downloaded CORA data, configured model paths, and optional caches.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--eq-jsonl", type=Path, default=None)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--base-cache-dir", type=Path, default=None)
    parser.add_argument("--source-cache-dir", type=Path, default=None)
    parser.add_argument("--external-cache-root", type=Path, default=None)
    parser.add_argument("--check-caches", action="store_true")
    parser.add_argument("--check-external", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = config.get("datasets") or {}
    runtime = config.get("runtime") or {}

    eq_jsonl = args.eq_jsonl or absolutize(datasets.get("cora_jsonl"))
    audio_dir = args.audio_dir or absolutize(datasets.get("audio_dir"))
    base_cache = args.base_cache_dir or absolutize(runtime.get("base_cache_dir"))
    source_cache = args.source_cache_dir or absolutize(runtime.get("source_cache_dir"))
    external_cache = args.external_cache_root or absolutize(runtime.get("external_cache_root"))

    rows: list[tuple[str, Path | None, bool, str]] = []
    add_check(rows, "CORA JSONL", eq_jsonl)
    add_check(rows, "CORA audio directory", audio_dir)

    for model in config.get("models", []):
        name = str(model.get("name"))
        if model.get("model_path"):
            add_check(rows, f"{name} model_path", absolutize(model.get("model_path")))
        if model.get("repo_path"):
            add_check(rows, f"{name} repo_path", absolutize(model.get("repo_path")))
        if model.get("checkpoint_path"):
            add_check(rows, f"{name} checkpoint_path", absolutize(model.get("checkpoint_path")))

    if args.check_external:
        for label, cfg in (config.get("external_retrievers") or {}).items():
            add_check(rows, f"{label} repo_path", absolutize(cfg.get("repo_path")))
            add_check(rows, f"{label} base_model_path", absolutize(cfg.get("base_model_path")), required="base_model_path" in cfg)
            add_check(rows, f"{label} checkpoint_path", absolutize(cfg.get("checkpoint_path")))

    if args.check_caches:
        add_check(rows, "base embedding cache dir", base_cache)
        add_check(rows, "source-caption embedding cache dir", source_cache)
        add_check(rows, "external embedding cache root", external_cache, required=False)

    ok = True
    for name, path, exists, required in rows:
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {name}: {path} ({required})")
        if required == "required" and not exists:
            ok = False
    print(f"\nOverall: {'OK' if ok else 'MISSING REQUIRED INPUTS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

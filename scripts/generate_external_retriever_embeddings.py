#!/usr/bin/env python3
"""Generate CORA-local embeddings for optional OEA/RobustCLAP retrievers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EQ_FORMS = ("key_phrase", "statement", "question", "command", "indirect")
ALL_TEXT_FORMS = ("source_caption", *EQ_FORMS)

OEA_BASE_MODEL = {
    "OEA-Nemo3B-Cl": "nvidia/omni-embed-nemotron-3b",
    "OEA-Nemo3B-AC": "nvidia/omni-embed-nemotron-3b",
    "OEA-Qwen3B-Cl": "Qwen/Qwen2.5-Omni-3B",
    "OEA-Qwen3B-AC": "Qwen/Qwen2.5-Omni-3B",
    "OEA-Qwen7B-Cl": "Qwen/Qwen2.5-Omni-7B",
    "OEA-Qwen7B-AC": "Qwen/Qwen2.5-Omni-7B",
}
OEA_CHECKPOINT_FILE = {
    "OEA-Nemo3B-Cl": "step_450_best.pt",
    "OEA-Nemo3B-AC": "step_400_best.pt",
    "OEA-Qwen3B-Cl": "step_40.pt",
    "OEA-Qwen3B-AC": "step_350.pt",
    "OEA-Qwen7B-Cl": "step_330.pt",
    "OEA-Qwen7B-AC": "step_300.pt",
}
ROBUSTCLAP_DEFAULT_FILE = "630k-audioset-best.pt"
ROBUSTCLAP_DEFAULT_FUSION_FILE = "630k-audioset-fusion-best.pt"


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_records(path: Path, max_items: int | None = None) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:max_items] if max_items is not None else records


def query_text(record: dict, form: str) -> str:
    if form == "source_caption":
        return str(record.get("source_caption", ""))
    return str(record.get("generated_queries", {}).get(form, ""))


def resolve_audio_path(record: dict, audio_dir: Path) -> Path:
    metadata = record.get("metadata", {}) or {}
    raw_candidates = [
        metadata.get("source_audio_path"),
        metadata.get("file_name"),
        record.get("audio_id"),
        f"{record.get('audio_id', '')}.wav" if record.get("audio_id") else None,
    ]
    dataset = str(record.get("dataset") or record.get("dataset_slug") or "")
    candidates: list[Path] = []
    for raw in raw_candidates:
        if not raw:
            continue
        p = Path(str(raw))
        if p.is_absolute():
            candidates.append(p)
        candidates.extend([audio_dir / p, audio_dir / dataset / p])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(p) for p in candidates[:6])
    raise FileNotFoundError(f"Could not resolve audio for {record.get('audio_id')}; tried: {tried}")


class Encoder:
    def __init__(self, encode_audio: Callable[[list[Path]], np.ndarray], encode_text: Callable[[list[str]], np.ndarray], metadata: dict):
        self.encode_audio = encode_audio
        self.encode_text = encode_text
        self.metadata = metadata


def build_oea_encoder(args: argparse.Namespace) -> Encoder:
    repo_root = Path(args.oea_repo_path).expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Omni-Embed-Audio repo missing: {repo_root}. Clone it manually and pass --oea-repo-path.")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from AudioRetrieval.models.omni_embed_adapter import OmniEmbedAdapter
    from AudioRetrieval.training.oea.train_omniembed_lora import ProjectionHead, attach_lora, resolve_text_hidden_size

    if not args.oea_base_model_path:
        raise ValueError("OEA requires --oea-base-model-path. Download the base model manually first.")
    if not args.oea_checkpoint_path:
        raise ValueError("OEA requires --oea-checkpoint-path. Download the adapter checkpoint manually first.")
    base_repo = str(Path(args.oea_base_model_path).expanduser().resolve())
    ckpt_path = Path(args.oea_checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"OEA checkpoint not found: {ckpt_path}")
    checkpoint_file = ckpt_path.name
    print(f"[oea] checkpoint={ckpt_path}", flush=True)
    print(f"[oea] base model={base_repo}", flush=True)
    adapter = OmniEmbedAdapter(repo_id=base_repo, device=args.device, passage_prefix="passage:", query_prefix="query:")

    lora_cfg = SimpleNamespace(
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "out_proj"],
    )
    print("[oea] attaching LoRA", flush=True)
    peft_model = attach_lora(adapter.get_underlying_model(), lora_cfg)
    if args.oea_device_map:
        adapter.model = peft_model
        adapter.model.eval()
    else:
        adapter.set_underlying_model(peft_model)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[oea] loading checkpoint from {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    peft_model.load_state_dict(ckpt["lora_state_dict"], strict=False)

    hidden_size = resolve_text_hidden_size(peft_model)
    audio_head = ProjectionHead(hidden_size, 512, 0.1).to(device).eval()
    text_head = ProjectionHead(hidden_size, 512, 0.1).to(device).eval()
    audio_head.load_state_dict(ckpt["audio_head"])
    text_head.load_state_dict(ckpt["text_head"])

    def encode_audio(paths: list[Path]) -> np.ndarray:
        outputs = []
        for batch in tqdm(list(batched(paths, args.audio_batch_size)), desc="[oea] audio", unit="batch"):
            raw = adapter.encode_audio([str(p) for p in batch], batch_size=len(batch))
            with torch.inference_mode():
                outputs.append(audio_head(torch.from_numpy(raw).to(device)).cpu().float().numpy())
        return l2(np.concatenate(outputs, axis=0))

    def encode_text(texts: list[str]) -> np.ndarray:
        outputs = []
        for batch in tqdm(list(batched(texts, args.text_batch_size)), desc="[oea] text", unit="batch"):
            raw = adapter.encode_text(batch, batch_size=len(batch))
            with torch.inference_mode():
                outputs.append(text_head(torch.from_numpy(raw).to(device)).cpu().float().numpy())
        return l2(np.concatenate(outputs, axis=0))

    return Encoder(
        encode_audio=encode_audio,
        encode_text=encode_text,
        metadata={"model_family": "oea", "checkpoint": args.oea_checkpoint, "base_model": base_repo, "checkpoint_file": checkpoint_file},
    )


def ensure_robustclap_bpe(src_root: Path) -> None:
    bpe_path = src_root / "laion_clap" / "clap_module" / "bpe_simple_vocab_16e6.txt.gz"
    if bpe_path.exists():
        return
    spec = importlib.util.find_spec("laion_clap")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("RobustCLAP fork is missing bpe_simple_vocab_16e6.txt.gz and installed laion_clap was not found.")
    candidate = Path(spec.origin).resolve().parent / "clap_module" / "bpe_simple_vocab_16e6.txt.gz"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find standard LAION CLAP BPE vocab at {candidate}")
    bpe_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, bpe_path)
    print(f"[robustclap] copied missing BPE vocab from installed laion_clap to {bpe_path}", flush=True)


def build_robustclap_encoder(args: argparse.Namespace) -> Encoder:
    repo_root = Path(args.robustclap_repo_path).expanduser().resolve()
    src_root = repo_root / "src"
    if not src_root.exists():
        raise FileNotFoundError(f"linguistic_robust_clap repo missing: {repo_root}. Clone it manually and pass --robustclap-repo-path.")
    ensure_robustclap_bpe(src_root)
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import torch
    import laion_clap
    from laion_clap.training.data import tokenizer as robustclap_tokenizer

    if not args.robustclap_checkpoint:
        raise ValueError("RobustCLAP requires --robustclap-checkpoint. Download the checkpoint manually first.")
    ckpt = Path(args.robustclap_checkpoint).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"RobustCLAP checkpoint not found: {ckpt}")
    checkpoint_source = str(ckpt)
    checkpoint_file = ckpt.name

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = laion_clap.CLAP_Module(
        enable_fusion=args.robustclap_enable_fusion,
        device=device,
        amodel=args.robustclap_amodel,
        tmodel=args.robustclap_tmodel,
    )
    original_torch_load = torch.load

    def compat_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = compat_torch_load
    try:
        model.load_ckpt(str(ckpt), verbose=False)
    finally:
        torch.load = original_torch_load

    def encode_audio(paths: list[Path]) -> np.ndarray:
        outputs = []
        for batch in tqdm(list(batched(paths, args.audio_batch_size)), desc="[robustclap] audio", unit="batch"):
            outputs.append(np.asarray(model.get_audio_embedding_from_filelist(x=[str(p) for p in batch], use_tensor=False)))
        return l2(np.concatenate(outputs, axis=0))

    def encode_text(texts: list[str]) -> np.ndarray:
        def tokenize(batch: list[str]):
            return robustclap_tokenizer(batch, tmodel=args.robustclap_tmodel)

        outputs = []
        for batch in tqdm(list(batched(texts, args.text_batch_size)), desc="[robustclap] text", unit="batch"):
            # The RobustCLAP/LAION wrapper can squeeze singleton tokenizer
            # outputs into 1D tensors. Duplicate then slice to keep batch shape.
            payload = batch if len(batch) > 1 else [batch[0], batch[0]]
            encoded = np.asarray(model.get_text_embedding(payload, tokenizer=tokenize, use_tensor=False))
            outputs.append(encoded[: len(batch)])
        return l2(np.concatenate(outputs, axis=0))

    return Encoder(
        encode_audio=encode_audio,
        encode_text=encode_text,
        metadata={
            "model_family": "robustclap",
            "checkpoint": checkpoint_source,
            "checkpoint_file": checkpoint_file,
            "enable_fusion": args.robustclap_enable_fusion,
            "amodel": args.robustclap_amodel,
            "tmodel": args.robustclap_tmodel,
            "text_tokenizer": args.robustclap_tmodel,
            "tokenizer_override": True,
        },
    )


def build_encoder(args: argparse.Namespace) -> Encoder:
    if args.model == "oea":
        return build_oea_encoder(args)
    if args.model == "robustclap":
        return build_robustclap_encoder(args)
    raise ValueError(args.model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CORA-local OEA/RobustCLAP embedding caches.")
    parser.add_argument("--model", choices=["oea", "robustclap"], required=True)
    parser.add_argument("--model-label", default=None, help="Output cache label. Defaults to --model.")
    parser.add_argument("--eq-jsonl", type=Path, default=PROJECT_ROOT / "data/cora/test/eq_by_clip.jsonl")
    parser.add_argument("--audio-dir", type=Path, default=PROJECT_ROOT / "data/cora/test/audio")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/eq_renew/external_retriever_embedding_cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audio-batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--oea-repo-path", type=Path, default=PROJECT_ROOT / "models_third_party/Omni-Embed-Audio")
    parser.add_argument("--oea-checkpoint", choices=sorted(OEA_BASE_MODEL), default="OEA-Qwen3B-Cl", help="Metadata label only; weights are loaded from --oea-checkpoint-path.")
    parser.add_argument("--oea-base-model-path", type=Path, default=None, help="Local OEA base model directory.")
    parser.add_argument("--oea-checkpoint-path", type=Path, default=None, help="Local OEA LoRA/head checkpoint file.")
    parser.add_argument("--oea-device-map", default=None, help="Passed to OmniEmbedAdapter, e.g. auto for multi-GPU sharding.")
    parser.add_argument("--oea-torch-dtype", default="auto", help="OEA base model dtype: auto, bfloat16, float16, float32.")
    parser.add_argument("--robustclap-repo-path", type=Path, default=PROJECT_ROOT / "models_third_party/linguistic_robust_clap")
    parser.add_argument("--robustclap-checkpoint", type=Path, default=None, help="Local RobustCLAP checkpoint file.")
    parser.add_argument("--robustclap-enable-fusion", action="store_true")
    parser.add_argument("--robustclap-amodel", default="HTSAT-tiny")
    parser.add_argument("--robustclap-tmodel", default="roberta")
    parser.add_argument("--allow-robustclap-default", action="store_true", help="Deprecated no-op kept for older commands.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label = args.model_label or args.model
    if not args.eq_jsonl.exists():
        raise FileNotFoundError(f"EQ jsonl missing: {args.eq_jsonl}. Prepare the CORA JSONL/audio files manually and pass --eq-jsonl/--audio-dir.")
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Audio dir missing: {args.audio_dir}. Prepare the CORA JSONL/audio files manually and pass --eq-jsonl/--audio-dir.")

    records = load_records(args.eq_jsonl, args.max_items)
    audio_paths = [resolve_audio_path(record, args.audio_dir) for record in records]
    print(f"[embed] model={args.model} label={label} records={len(records)}", flush=True)
    encoder = build_encoder(args)

    out_dir = args.output_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[embed] encoding audio: {len(audio_paths)}", flush=True)
    audio = encoder.encode_audio(audio_paths)
    text_arrays = {}
    for form in ALL_TEXT_FORMS:
        print(f"[embed] encoding text form={form}", flush=True)
        text_arrays[form] = encoder.encode_text([query_text(record, form) for record in records])

    np.save(out_dir / "audio_embeddings.npy", audio)
    np.savez_compressed(out_dir / "text_embeddings.npz", **text_arrays)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "model_label": label,
        "num_items": len(records),
        "eq_jsonl": str(args.eq_jsonl),
        "audio_dir": str(args.audio_dir),
        "text_forms": list(ALL_TEXT_FORMS),
        **encoder.metadata,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

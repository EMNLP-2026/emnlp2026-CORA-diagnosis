#!/usr/bin/env python3
"""Run Boundary-Margin Token Saliency diagnostics for CORA retrievers.

This replaces the old leave-one-token-group-out deletion diagnostic. The EQ
query is kept unchanged; saliency is computed on the target-vs-hard-negative
boundary margin M(q) = cos(t(q), a_target) - cos(t(q), a_hn).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clap_eval.models import get_model  # noqa: E402

BASE_MODELS = ("laion", "msclap", "m2d", "mga")
SUPPORTED_GRADIENT_MODELS = BASE_MODELS
EQ_FORMS = ("key_phrase", "statement", "question", "command", "indirect")
DEFAULT_QUERY_FORMS = ("statement", "question", "command", "indirect")
ALL_TEXT_FORMS = ("source_caption", *EQ_FORMS)
BASE_CACHE = PROJECT_ROOT / "results/eq_renew/vanilla_multiclap_eq_embedding_cache"
SOURCE_CACHE = PROJECT_ROOT / "results/eq_renew/single_model_invariance_embedding_cache"
EQ_JSONL = PROJECT_ROOT / "data/cora/test/eq_by_clip.jsonl"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[?]")
INTENT_TOKENS = {
    "find",
    "search",
    "retrieve",
    "locate",
    "identify",
    "get",
    "hear",
    "heard",
    "listen",
    "audio",
    "clip",
    "sound",
    "sounds",
    "recording",
    "track",
}
FUNCTION_FORM_TOKENS = {
    "?",
    "a",
    "an",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "and",
    "or",
    "but",
    "can",
    "could",
    "would",
    "will",
    "you",
    "please",
    "i",
    "i'm",
    "im",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "do",
    "does",
    "did",
    "where",
    "what",
    "which",
    "that",
    "there",
    "this",
    "these",
    "those",
    "looking",
    "look",
    "like",
    "someone",
    "something",
    "anything",
    "some",
    "any",
    "kind",
    "type",
}
STOP_FOR_CONTENT = FUNCTION_FORM_TOKENS | INTENT_TOKENS | {"as", "into", "about", "over", "under", "near", "while", "then"}
TOKEN_GROUPS = ("acoustic_content", "query_intent", "function_form")
ALL_GROUPS = (*TOKEN_GROUPS, "other")
OUTCOMES = ("retained", "hit_drop")
OLD_ABLATION_FILES = (
    "external_token_ablation_detail.csv",
    "external_token_ablation_summary.csv",
    "external_token_ablation_regression.csv",
)


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern} in {root}; found {len(matches)}")
    return matches[0]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def model_config(config: dict, model_name: str) -> dict:
    for item in config.get("models", []):
        if str(item.get("name", "")).lower() == model_name:
            cfg = dict(item)
            for key in ("repo_path", "checkpoint_path"):
                value = cfg.get(key)
                if value:
                    p = Path(value)
                    cfg[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
            return cfg
    raise ValueError(f"Missing model config for {model_name}")


def find_source_cache(source_cache: Path, model_label: str, n: int) -> Path:
    matches = []
    for path in sorted(source_cache.glob(f"*text_{model_label}_*.npz")):
        with np.load(path) as z:
            if "source_caption" in z.files and z["source_caption"].shape[0] == n:
                matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source_caption cache for {model_label}, n={n}; found {len(matches)}")
    return matches[0]


def load_base_cache(core_module, args: argparse.Namespace, model_label: str) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray], dict]:
    records = core_module.load_jsonl(args.eq_jsonl)
    audio_path = find_one(args.base_cache_dir, f"*audio_{model_label}_*.npy")
    text_path = find_one(args.base_cache_dir, f"*text_{model_label}_*.npz")
    audio = l2(np.load(audio_path))
    with np.load(text_path) as z:
        missing = [form for form in EQ_FORMS if form not in z.files]
        if missing:
            raise ValueError(f"{text_path} missing forms: {missing}")
        text = {form: l2(z[form]) for form in EQ_FORMS}
    source_path = find_source_cache(args.source_cache_dir, model_label, len(audio))
    with np.load(source_path) as z:
        text["source_caption"] = l2(z["source_caption"])
        if "statement" in z.files:
            delta = float(np.max(np.abs(l2(z["statement"]) - text["statement"])))
            if delta > 1e-4:
                raise ValueError(f"Cache order mismatch for {model_label}: statement delta={delta}")
    if len(records) != len(audio):
        raise ValueError(f"Cache length mismatch for {model_label}: records={len(records)} audio={len(audio)}")
    for form in ALL_TEXT_FORMS:
        if len(text[form]) != len(records):
            raise ValueError(f"Text length mismatch for {model_label}/{form}: {len(text[form])} vs {len(records)}")
    return records, audio, text, {"family": "base_clap"}


def infer_family(model_label: str) -> str:
    lower = model_label.lower()
    if "robust" in lower or "clap" in lower:
        return "robustclap"
    if "oea" in lower or "omni" in lower:
        return "oea"
    raise ValueError(f"Cannot infer model family from cache label {model_label}; use labels containing oea or robustclap.")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def token_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(text)]


def content_vocab(source_caption: str) -> set[str]:
    return {tok.lower() for tok in tokenize(source_caption) if tok.lower() not in STOP_FOR_CONTENT and tok != "?"}


def classify_surface_token(token: str, source_content: set[str]) -> str:
    low = token.lower()
    if low in INTENT_TOKENS:
        return "query_intent"
    if low in FUNCTION_FORM_TOKENS:
        return "function_form"
    if low in source_content or (low not in STOP_FOR_CONTENT and low != "?"):
        return "acoustic_content"
    return "other"


def classify_query_spans(query: str, source_caption: str) -> list[dict[str, Any]]:
    source_content = content_vocab(source_caption)
    spans = []
    for start, end, token in token_spans(query):
        spans.append({"start": start, "end": end, "token": token, "group": classify_surface_token(token, source_content)})
    return spans


def clean_model_token(token: str) -> str:
    token = str(token)
    token = token.replace("</w>", "")
    token = token.replace("##", "")
    token = token.replace("Ġ", "")
    token = token.replace("▁", "")
    token = token.strip()
    return token


def group_for_offsets(start: int, end: int, spans: list[dict[str, Any]], fallback_token: str) -> tuple[str, str]:
    if end <= start:
        return "special", ""
    best = None
    best_overlap = 0
    for span in spans:
        overlap = max(0, min(end, int(span["end"])) - max(start, int(span["start"])))
        if overlap > best_overlap:
            best_overlap = overlap
            best = span
    if best is not None and best_overlap > 0:
        return str(best["group"]), str(best["token"])
    cleaned = clean_model_token(fallback_token)
    if not cleaned:
        return "special", ""
    return "other", cleaned


def groups_from_wordpieces(tokens: list[str], special_mask: list[int], source_caption: str) -> list[tuple[str, str]]:
    source_content = content_vocab(source_caption)
    out = []
    current_group = "other"
    current_surface = ""
    for token, is_special in zip(tokens, special_mask):
        if is_special:
            out.append(("special", ""))
            continue
        cleaned = clean_model_token(token)
        if not cleaned:
            out.append(("special", ""))
            continue
        if str(token).startswith("##") and current_group != "special":
            out.append((current_group, cleaned))
            current_surface += cleaned
            continue
        current_group = classify_surface_token(cleaned, source_content)
        current_surface = cleaned
        out.append((current_group, cleaned))
    return out


def device_string(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    return requested if torch.cuda.is_available() else "cpu"


def first_embedding_module(module):
    import torch

    for name, child in module.named_modules():
        if isinstance(child, torch.nn.Embedding):
            return name, child
    raise RuntimeError(f"No torch.nn.Embedding module found in {type(module)}")


@dataclass
class SaliencyOutput:
    text_embedding: np.ndarray
    boundary_margin: float
    target_score: float
    hard_negative_score: float
    tokens: list[dict[str, Any]]


class EmbeddingCapture:
    def __init__(self, module):
        self.embedding = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            raise TypeError("Tuple embedding outputs are not supported by this saliency hook")
        captured = output.detach().requires_grad_(True)
        captured.retain_grad()
        self.embedding = captured
        return captured

    def close(self) -> None:
        self.handle.remove()


class BoundaryMarginSaliencyRunner:
    def __init__(self, model_label: str, args: argparse.Namespace):
        self.model_label = model_label
        self.args = args
        self.device = device_string(args.device)
        self.config = load_config(args.config)
        if model_label not in SUPPORTED_GRADIENT_MODELS:
            raise NotImplementedError(
                f"Gradient saliency is implemented for {SUPPORTED_GRADIENT_MODELS}; {model_label} uses cached embeddings only."
            )
        self.wrapper = get_model(model_label, model_config(self.config, model_label), self.device)
        self._prepare_model()

    def _prepare_model(self) -> None:
        import torch

        model = getattr(self.wrapper, "model", None)
        if hasattr(model, "eval"):
            model.eval()
        if self.model_label == "msclap":
            self.wrapper.model.clap.eval()
            self.wrapper.model.clap.to(self.device)
        if self.model_label == "m2d" and not hasattr(self.wrapper.model, "text_encoder"):
            self.wrapper.model.get_clap_text_encoder()
        for param in self._torch_root().parameters():
            param.requires_grad_(False)
        torch.set_grad_enabled(True)

    def _torch_root(self):
        if self.model_label == "msclap":
            return self.wrapper.model.clap
        return self.wrapper.model

    def _zero_grad(self) -> None:
        root = self._torch_root()
        if hasattr(root, "zero_grad"):
            root.zero_grad(set_to_none=True)

    def _target_tensors(self, target_audio: np.ndarray, hard_negative_audio: np.ndarray):
        import torch

        target = torch.as_tensor(target_audio, dtype=torch.float32, device=self.device)
        hard_negative = torch.as_tensor(hard_negative_audio, dtype=torch.float32, device=self.device)
        return target, hard_negative

    def _finish_saliency(
        self,
        text_embedding,
        target_audio: np.ndarray,
        hard_negative_audio: np.ndarray,
        capture: EmbeddingCapture,
        token_meta: list[dict[str, Any]],
    ) -> SaliencyOutput:
        import torch
        import torch.nn.functional as F

        if text_embedding.ndim == 1:
            text_embedding = text_embedding.unsqueeze(0)
        text_embedding = F.normalize(text_embedding.float(), p=2, dim=-1)
        target, hard_negative = self._target_tensors(target_audio, hard_negative_audio)
        target_score = torch.sum(text_embedding[0] * target)
        hard_negative_score = torch.sum(text_embedding[0] * hard_negative)
        margin = target_score - hard_negative_score
        margin.backward()
        if capture.embedding is None or capture.embedding.grad is None:
            raise RuntimeError(f"No embedding gradient captured for {self.model_label}")
        raw = (capture.embedding.detach()[0] * capture.embedding.grad.detach()[0]).sum(dim=-1).float().cpu().numpy()
        total_abs = float(np.sum(np.abs(raw)))
        denom = total_abs + 1e-12
        rows = []
        for idx, meta in enumerate(token_meta):
            if idx >= len(raw):
                break
            group = str(meta.get("token_group", "other"))
            if group == "special":
                continue
            value = float(raw[idx])
            rows.append(
                {
                    **meta,
                    "saliency": value,
                    "normalized_saliency": value / denom,
                    "positive_saliency_share": max(value, 0.0) / denom,
                    "negative_saliency_share": max(-value, 0.0) / denom,
                    "absolute_saliency_share": abs(value) / denom,
                }
            )
        return SaliencyOutput(
            text_embedding=text_embedding.detach().cpu().numpy()[0],
            boundary_margin=float(margin.detach().cpu()),
            target_score=float(target_score.detach().cpu()),
            hard_negative_score=float(hard_negative_score.detach().cpu()),
            tokens=rows,
        )

    def _meta_from_offsets(self, text: str, source_caption: str, input_ids, offsets, attention_mask, tokenizer) -> list[dict[str, Any]]:
        spans = classify_query_spans(text, source_caption)
        tokens = tokenizer.convert_ids_to_tokens([int(x) for x in input_ids])
        out = []
        for idx, (token, offset, keep) in enumerate(zip(tokens, offsets, attention_mask)):
            if int(keep) == 0:
                out.append({"token_index": idx, "model_token": token, "token_surface": "", "token_group": "special"})
                continue
            start, end = int(offset[0]), int(offset[1])
            group, surface = group_for_offsets(start, end, spans, token)
            out.append(
                {
                    "token_index": idx,
                    "model_token": token,
                    "token_surface": text[start:end] if end > start else surface,
                    "char_start": start if end > start else -1,
                    "char_end": end if end > start else -1,
                    "token_group": group,
                }
            )
        return out

    def _meta_from_wordpieces(self, source_caption: str, input_ids, special_mask, tokenizer) -> list[dict[str, Any]]:
        tokens = tokenizer.convert_ids_to_tokens([int(x) for x in input_ids])
        groups = groups_from_wordpieces(tokens, [int(x) for x in special_mask], source_caption)
        return [
            {
                "token_index": idx,
                "model_token": token,
                "token_surface": surface,
                "char_start": -1,
                "char_end": -1,
                "token_group": group,
            }
            for idx, (token, (group, surface)) in enumerate(zip(tokens, groups))
        ]

    def saliency(self, text: str, source_caption: str, target_audio: np.ndarray, hard_negative_audio: np.ndarray) -> SaliencyOutput:
        if self.model_label == "laion":
            return self._saliency_laion(text, source_caption, target_audio, hard_negative_audio)
        if self.model_label == "msclap":
            return self._saliency_msclap(text, source_caption, target_audio, hard_negative_audio)
        if self.model_label == "m2d":
            return self._saliency_m2d(text, source_caption, target_audio, hard_negative_audio)
        if self.model_label == "mga":
            return self._saliency_mga(text, source_caption, target_audio, hard_negative_audio)
        raise NotImplementedError(self.model_label)

    def _saliency_laion(self, text: str, source_caption: str, target_audio: np.ndarray, hard_negative_audio: np.ndarray) -> SaliencyOutput:
        self._zero_grad()
        tokenizer = self.wrapper.processor.tokenizer
        encoded = tokenizer([text], return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")[0].cpu().numpy()
        input_ids = encoded["input_ids"][0].cpu().numpy()
        attention = encoded["attention_mask"][0].cpu().numpy()
        token_meta = self._meta_from_offsets(text, source_caption, input_ids, offsets, attention, tokenizer)
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        _, module = first_embedding_module(self.wrapper.model.text_model.embeddings)
        capture = EmbeddingCapture(module)
        try:
            outputs = self.wrapper.model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                outputs = outputs.pooler_output
            return self._finish_saliency(outputs, target_audio, hard_negative_audio, capture, token_meta)
        finally:
            capture.close()

    def _saliency_msclap(self, text: str, source_caption: str, target_audio: np.ndarray, hard_negative_audio: np.ndarray) -> SaliencyOutput:
        self._zero_grad()
        tokenizer = self.wrapper.model.tokenizer
        encoded_offsets = tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
            return_offsets_mapping=True,
        )
        input_ids = encoded_offsets["input_ids"][0].cpu().numpy()
        attention = encoded_offsets["attention_mask"][0].cpu().numpy()
        offsets = encoded_offsets["offset_mapping"][0].cpu().numpy()
        token_meta = self._meta_from_offsets(text, source_caption, input_ids, offsets, attention, tokenizer)
        preprocessed = self.wrapper.model.preprocess_text([text])
        preprocessed = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in preprocessed.items()}
        module = self.wrapper.model.clap.caption_encoder.base.wte
        capture = EmbeddingCapture(module)
        try:
            outputs = self.wrapper.model.clap.caption_encoder(preprocessed)
            return self._finish_saliency(outputs, target_audio, hard_negative_audio, capture, token_meta)
        finally:
            capture.close()

    def _saliency_m2d(self, text: str, source_caption: str, target_audio: np.ndarray, hard_negative_audio: np.ndarray) -> SaliencyOutput:
        self._zero_grad()
        if not hasattr(self.wrapper.model, "text_encoder"):
            self.wrapper.model.get_clap_text_encoder()
        text_encoder = self.wrapper.model.text_encoder
        tokenizer = text_encoder.tokenizer
        encoded = tokenizer(
            [text],
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"][0].cpu().numpy()
        input_ids = encoded["input_ids"][0].cpu().numpy()
        attention = encoded["attention_mask"][0].cpu().numpy()
        token_meta = self._meta_from_offsets(text, source_caption, input_ids, offsets, attention, tokenizer)
        _, module = first_embedding_module(text_encoder)
        capture = EmbeddingCapture(module)
        try:
            outputs = self.wrapper.model.encode_clap_text([text])
            return self._finish_saliency(outputs, target_audio, hard_negative_audio, capture, token_meta)
        finally:
            capture.close()

    def _saliency_mga(self, text: str, source_caption: str, target_audio: np.ndarray, hard_negative_audio: np.ndarray) -> SaliencyOutput:
        self._zero_grad()
        tokenizer = self.wrapper.model.text_encoder.tokenizer
        encoded = tokenizer(
            [text],
            padding="longest",
            truncation=True,
            max_length=30,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        input_ids = encoded["input_ids"][0].cpu().numpy()
        special_mask = encoded["special_tokens_mask"][0].cpu().numpy()
        token_meta = self._meta_from_wordpieces(source_caption, input_ids, special_mask, tokenizer)
        module = self.wrapper.model.text_encoder.text_encoder.embeddings.word_embeddings
        capture = EmbeddingCapture(module)
        try:
            _, word_embeds, attn_mask = self.wrapper.model.encode_text([text])
            outputs = self.wrapper.model.msc(word_embeds, self.wrapper.model.codebook, attn_mask)
            return self._finish_saliency(outputs, target_audio, hard_negative_audio, capture, token_meta)
        finally:
            capture.close()


def load_inputs(model_label: str, args: argparse.Namespace, core_module):
    if model_label in BASE_MODELS:
        return load_base_cache(core_module, args, model_label)
    return core_module.load_cache(args.cache_root, model_label)


def sample_balanced(frame: pd.DataFrame, query_forms: list[str], max_per_cell: int, seed: int) -> pd.DataFrame:
    eligible = frame[
        (frame["source_hit5"] == 1)
        & (frame["outcome"].isin(OUTCOMES))
        & (frame["query_form"].isin(query_forms))
    ].copy()
    if max_per_cell <= 0:
        return eligible.reset_index(drop=True)
    pieces = []
    for _, group in eligible.groupby(["dataset", "query_form", "outcome"], sort=False):
        n = min(max_per_cell, len(group))
        pieces.append(group.sample(n=n, random_state=seed) if len(group) > n else group)
    return pd.concat(pieces, ignore_index=True) if pieces else eligible.iloc[0:0]


def group_shares(token_rows: list[dict[str, Any]]) -> dict[str, float]:
    shares = {f"{group}_positive_share": 0.0 for group in ALL_GROUPS}
    shares.update({f"{group}_negative_share": 0.0 for group in ALL_GROUPS})
    shares.update({f"{group}_signed_share": 0.0 for group in ALL_GROUPS})
    shares.update({f"{group}_absolute_share": 0.0 for group in ALL_GROUPS})
    for row in token_rows:
        group = str(row.get("token_group", "other"))
        if group not in ALL_GROUPS:
            group = "other"
        shares[f"{group}_positive_share"] += float(row.get("positive_saliency_share", 0.0))
        shares[f"{group}_negative_share"] += float(row.get("negative_saliency_share", 0.0))
        shares[f"{group}_signed_share"] += float(row.get("normalized_saliency", 0.0))
        shares[f"{group}_absolute_share"] += float(row.get("absolute_saliency_share", 0.0))
    return shares


def pair_identity(row: pd.Series) -> dict[str, Any]:
    keep = [
        "model",
        "dataset",
        "audio_id",
        "row_index",
        "query_form",
        "form_group",
        "outcome",
        "source_hit5",
        "eq_hit5",
        "retained",
        "hit_drop",
        "source_rank",
        "eq_rank",
        "rank_drop",
        "query_move_cosdist",
        "query_movement_norm",
        "target_align_delta",
        "target_hn_margin_eq",
        "union_top5_margin_delta",
        "embedding_margin_delta",
        "hn_minus_target_delta",
        "p_audio",
        "p_boundary",
        "union_hn_index",
        "union_hn_audio_id",
    ]
    return {key: row[key] for key in keep if key in row.index}


def run_model(model_label: str, args: argparse.Namespace, core_module) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if model_label not in SUPPORTED_GRADIENT_MODELS:
        status = {"model": model_label, "status": "skipped", "note": "gradient saliency adapter is not implemented for this cached retriever"}
        return pd.DataFrame(), pd.DataFrame(), status
    records, audio, text, metadata = load_inputs(model_label, args, core_module)
    pairwise = core_module.compute_pairwise(model_label, records, audio, text, args.top_k)
    selected = sample_balanced(pairwise, args.query_forms, args.max_per_cell, args.seed)
    if selected.empty:
        status = {"model": model_label, "status": "skipped", "note": "no retained/hit_drop source-hit pairs after filtering"}
        return pd.DataFrame(), pd.DataFrame(), status

    runner = BoundaryMarginSaliencyRunner(model_label, args)
    pair_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    failures = 0
    for pair_id, (_, row) in enumerate(selected.iterrows()):
        record_index = int(row["row_index"])
        hn_index = int(row["union_hn_index"])
        form = str(row["query_form"])
        source_caption = core_module.query_text(records[record_index], "source_caption")
        eq_query = core_module.query_text(records[record_index], form)
        try:
            sal = runner.saliency(eq_query, source_caption, audio[record_index], audio[hn_index])
        except Exception as exc:
            failures += 1
            if args.keep_going:
                continue
            raise RuntimeError(f"Saliency failed for {model_label} row={record_index} form={form}: {exc}") from exc
        base = pair_identity(row)
        base.update(
            {
                "pair_id": f"{model_label}:{record_index}:{form}",
                "source_caption": source_caption,
                "eq_query": eq_query,
                "saliency_target_score": sal.target_score,
                "saliency_hard_negative_score": sal.hard_negative_score,
                "saliency_boundary_margin": sal.boundary_margin,
                "cache_boundary_margin_eq": float(row.get("target_hn_margin_eq", np.nan)),
                "saliency_cache_margin_delta_abs": abs(sal.boundary_margin - float(row.get("target_hn_margin_eq", np.nan)))
                if not pd.isna(row.get("target_hn_margin_eq", np.nan))
                else float("nan"),
                "token_count": len(sal.tokens),
            }
        )
        base.update(group_shares(sal.tokens))
        pair_rows.append(base)
        for token in sal.tokens:
            token_rows.append({**base, **token})
        if args.progress_every and (pair_id + 1) % args.progress_every == 0:
            print(f"[{model_label}] saliency pairs {pair_id + 1}/{len(selected)}", flush=True)

    status = {
        "model": model_label,
        "status": "completed",
        "selected_pairs": int(len(selected)),
        "completed_pairs": int(len(pair_rows)),
        "failed_pairs": int(failures),
        "metadata": metadata,
    }
    return pd.DataFrame(pair_rows), pd.DataFrame(token_rows), status


def summarize_groups(pair_detail: pd.DataFrame) -> pd.DataFrame:
    if pair_detail.empty:
        return pd.DataFrame()
    cols = [
        "acoustic_content_positive_share",
        "acoustic_content_negative_share",
        "function_form_negative_share",
        "query_intent_negative_share",
        "function_form_positive_share",
        "query_intent_positive_share",
        "other_positive_share",
        "other_negative_share",
        "saliency_boundary_margin",
        "saliency_cache_margin_delta_abs",
        "token_count",
    ]
    grouped = pair_detail.groupby(["model", "outcome"], sort=False)
    summary = grouped.agg(n_pairs=("pair_id", "nunique"), **{col: (col, "mean") for col in cols if col in pair_detail.columns}).reset_index()
    return summary


def summarize_by_form(pair_detail: pd.DataFrame) -> pd.DataFrame:
    if pair_detail.empty:
        return pd.DataFrame()
    cols = [
        "acoustic_content_positive_share",
        "acoustic_content_negative_share",
        "function_form_negative_share",
        "query_intent_negative_share",
        "saliency_boundary_margin",
        "token_count",
    ]
    grouped = pair_detail.groupby(["model", "query_form", "outcome"], sort=False)
    return grouped.agg(n_pairs=("pair_id", "nunique"), **{col: (col, "mean") for col in cols if col in pair_detail.columns}).reset_index()


def cv_auc(frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> tuple[float, str, int]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:  # pragma: no cover
        return float("nan"), f"skipped: {exc}", 0

    cols = numeric_features + categorical_features + ["hit_drop", "audio_id"]
    data = frame[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_features + ["hit_drop"]).copy()
    if len(data) < 20 or data["hit_drop"].nunique() < 2:
        return float("nan"), "skipped: insufficient rows/classes", len(data)
    y = data["hit_drop"].astype(int)
    class_min = int(y.value_counts().min())
    group_count = int(data["audio_id"].nunique())
    n_splits = min(5, class_min, group_count)
    if n_splits < 2:
        return float("nan"), "skipped: insufficient class/group counts", len(data)
    preprocess = ColumnTransformer(
        [("num", StandardScaler(), numeric_features), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)]
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipeline = Pipeline([("preprocess", preprocess), ("clf", clf)])
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=17)
        pred = cross_val_predict(
            pipeline,
            data[numeric_features + categorical_features],
            y,
            groups=data["audio_id"],
            cv=cv,
            method="predict_proba",
        )[:, 1]
        return float(roc_auc_score(y, pred)), f"{n_splits}-fold StratifiedGroupKFold grouped by audio_id", len(data)
    except Exception as exc:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17)
        pred = cross_val_predict(pipeline, data[numeric_features + categorical_features], y, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, pred)), f"{n_splits}-fold StratifiedKFold fallback: {exc}", len(data)


def prediction_summary(pair_detail: pd.DataFrame) -> pd.DataFrame:
    if pair_detail.empty:
        return pd.DataFrame()
    feature_sets = [
        ("fixed_effects_only", [], ["model", "dataset", "query_form"]),
        ("plus_query_movement", ["query_move_cosdist", "query_movement_norm", "target_align_delta"], ["model", "dataset", "query_form"]),
        (
            "plus_token_group_saliency",
            [
                "acoustic_content_positive_share",
                "acoustic_content_negative_share",
                "function_form_negative_share",
                "query_intent_negative_share",
                "function_form_positive_share",
                "query_intent_positive_share",
            ],
            ["model", "dataset", "query_form"],
        ),
        (
            "plus_boundary_features",
            ["query_move_cosdist", "query_movement_norm", "target_align_delta", "target_hn_margin_eq", "union_top5_margin_delta", "p_boundary"],
            ["model", "dataset", "query_form"],
        ),
        (
            "boundary_plus_token_saliency",
            [
                "query_move_cosdist",
                "query_movement_norm",
                "target_align_delta",
                "target_hn_margin_eq",
                "union_top5_margin_delta",
                "p_boundary",
                "acoustic_content_positive_share",
                "acoustic_content_negative_share",
                "function_form_negative_share",
                "query_intent_negative_share",
                "function_form_positive_share",
                "query_intent_positive_share",
            ],
            ["model", "dataset", "query_form"],
        ),
    ]
    rows = []
    for model, group in pair_detail.groupby("model", sort=False):
        for name, numeric, categorical in feature_sets:
            auc, note, n = cv_auc(group, numeric, categorical)
            rows.append({"model": model, "feature_set": name, "n_pairs": n, "auc": auc, "note": note})
    for name, numeric, categorical in feature_sets:
        auc, note, n = cv_auc(pair_detail, numeric, categorical)
        rows.append({"model": "pooled", "feature_set": name, "n_pairs": n, "auc": auc, "note": note})
    return pd.DataFrame(rows)


def permutation_sanity(token_detail: pd.DataFrame, seed: int) -> pd.DataFrame:
    if token_detail.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    real_pairs = []
    perm_pairs = []
    for pair_id, group in token_detail.groupby("pair_id", sort=False):
        real_pairs.append({"pair_id": pair_id, **group.iloc[0][["model", "dataset", "audio_id", "query_form", "outcome", "hit_drop"]].to_dict(), **group_shares(group.to_dict("records"))})
        records = group.to_dict("records")
        labels = [str(row.get("token_group", "other")) for row in records]
        rng.shuffle(labels)
        permuted = []
        for row, label in zip(records, labels):
            item = dict(row)
            item["token_group"] = label
            permuted.append(item)
        perm_pairs.append({"pair_id": pair_id, **group.iloc[0][["model", "dataset", "audio_id", "query_form", "outcome", "hit_drop"]].to_dict(), **group_shares(permuted)})
    real = pd.DataFrame(real_pairs)
    perm = pd.DataFrame(perm_pairs)
    metrics = [
        "acoustic_content_positive_share",
        "acoustic_content_negative_share",
        "function_form_negative_share",
        "query_intent_negative_share",
    ]
    rows = []
    for (model, outcome), real_group in real.groupby(["model", "outcome"], sort=False):
        perm_group = perm[(perm["model"] == model) & (perm["outcome"] == outcome)]
        for metric in metrics:
            rows.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "metric": metric,
                    "real_mean": float(real_group[metric].mean()),
                    "permuted_mean": float(perm_group[metric].mean()),
                    "real_minus_permuted": float(real_group[metric].mean() - perm_group[metric].mean()),
                    "n_pairs": int(len(real_group)),
                }
            )
    return pd.DataFrame(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df[columns].head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    pair_detail: pd.DataFrame,
    group_summary: pd.DataFrame,
    form_summary: pd.DataFrame,
    prediction: pd.DataFrame,
    sanity: pd.DataFrame,
    status: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    report = [
        "# Boundary-Margin Token Saliency Report",
        "",
        "This Action 3 diagnostic replaces the old leave-one-token-group-out deletion ablation. It keeps the original EQ query unchanged and computes token-level Gradient x Input saliency for M(q) = cos(t(q), a_target) - cos(t(q), a_hn).",
        "",
        f"Models requested: `{', '.join(args.models)}`",
        f"Query forms: `{', '.join(args.query_forms)}`",
        f"Sampling: max `{args.max_per_cell}` per dataset x query_form x outcome cell; outcomes are retained and hit_drop; source-caption hit@5 pairs only.",
        f"Pair rows: `{len(pair_detail)}`",
        "",
        "## Run Status",
        "",
        markdown_table(status, [col for col in ["model", "status", "selected_pairs", "completed_pairs", "failed_pairs", "note"] if col in status.columns], max_rows=40),
        "",
        "## Main Table 1: Token-Group Attribution by Outcome",
        "",
        markdown_table(
            group_summary,
            [
                "model",
                "outcome",
                "n_pairs",
                "acoustic_content_positive_share",
                "acoustic_content_negative_share",
                "function_form_negative_share",
                "query_intent_negative_share",
                "saliency_boundary_margin",
            ],
            max_rows=80,
        ),
        "",
        "## Form-Level Check",
        "",
        markdown_table(
            form_summary,
            [
                "model",
                "query_form",
                "outcome",
                "n_pairs",
                "acoustic_content_positive_share",
                "function_form_negative_share",
                "query_intent_negative_share",
            ],
            max_rows=80,
        ),
        "",
        "## Main Table 2: Hit-Drop Prediction",
        "",
        markdown_table(prediction, ["model", "feature_set", "n_pairs", "auc", "note"], max_rows=120),
        "",
        "## Random Group-Label Sanity Check",
        "",
        markdown_table(sanity, ["model", "outcome", "metric", "real_mean", "permuted_mean", "real_minus_permuted", "n_pairs"], max_rows=120),
        "",
        "## Interpretation Contract",
        "",
        "Positive token saliency means the token locally supports the target over the selected hard negative. Negative token saliency means the token locally supports the hard negative over the target. Group shares are normalized within each unchanged query by the sum of absolute token saliency. These are local gradient attributions of the text encoder boundary margin, not causal token-deletion effects.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Boundary-Margin Token Saliency diagnostics for CORA retrievers.")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "results/eq_renew/external_retriever_embedding_cache")
    parser.add_argument("--base-cache-dir", type=Path, default=BASE_CACHE)
    parser.add_argument("--source-cache-dir", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--eq-jsonl", type=Path, default=EQ_JSONL)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--models", nargs="+", default=list(BASE_MODELS), help="Models to analyze. Exact gradient saliency is currently implemented for base CLAP models.")
    parser.add_argument("--query-forms", nargs="+", choices=list(EQ_FORMS), default=list(DEFAULT_QUERY_FORMS))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments/token_saliency_helper_debug")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-per-cell", type=int, default=10, help="Sample per dataset/form/outcome cell. Use 0 for all eligible rows.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-going", action="store_true", help="Skip individual failed saliency rows instead of failing the run.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--replace-output", action="store_true", help="Remove legacy deletion-ablation CSV outputs before writing saliency artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    core = load_module("external_core_diagnostic", PROJECT_ROOT / "scripts/run_external_core_diagnostic.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.replace_output:
        for name in OLD_ABLATION_FILES:
            old = args.output_dir / name
            if old.exists():
                old.unlink()

    pair_frames = []
    token_frames = []
    statuses = []
    for model_label in args.models:
        print(f"[saliency] starting {model_label}", flush=True)
        pair_detail, token_detail, status = run_model(model_label, args, core)
        pair_frames.append(pair_detail)
        token_frames.append(token_detail)
        statuses.append(status)
        print(f"[saliency] {model_label}: {status}", flush=True)

    pair_all = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    token_all = pd.concat(token_frames, ignore_index=True) if token_frames else pd.DataFrame()
    status_df = pd.DataFrame(statuses)
    if pair_all.empty:
        status_df.to_csv(args.output_dir / "boundary_margin_token_saliency_status.csv", index=False)
        raise SystemExit("No saliency rows produced. Check supported models, caches, and retained/hit_drop availability.")

    group_summary = summarize_groups(pair_all)
    form_summary = summarize_by_form(pair_all)
    prediction = prediction_summary(pair_all)
    sanity = permutation_sanity(token_all, args.seed)

    pair_all.to_csv(args.output_dir / "boundary_margin_token_saliency_pairs.csv", index=False)
    token_all.to_csv(args.output_dir / "boundary_margin_token_saliency_tokens.csv", index=False)
    group_summary.to_csv(args.output_dir / "boundary_margin_token_saliency_group_summary.csv", index=False)
    form_summary.to_csv(args.output_dir / "boundary_margin_token_saliency_form_summary.csv", index=False)
    prediction.to_csv(args.output_dir / "boundary_margin_hitdrop_prediction.csv", index=False)
    sanity.to_csv(args.output_dir / "boundary_margin_permutation_sanity.csv", index=False)
    status_df.to_csv(args.output_dir / "boundary_margin_token_saliency_status.csv", index=False)
    write_report(args.output_dir, pair_all, group_summary, form_summary, prediction, sanity, status_df, args)
    print(f"Wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

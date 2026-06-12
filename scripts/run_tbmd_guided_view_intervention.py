#!/usr/bin/env python3
"""TBMD-guided EQ view intervention.

This is a controlled, target-aware oracle experiment. It asks whether the
diagnostic axis has actionable content: if one could choose among the five EQ
views using TBMD@5, how much target RankDrop is reduced compared with choosing
the closest text-space view.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


ROOT = Path(os.environ.get("CORA_ROOT") or Path(__file__).resolve().parents[1])
OUT = ROOT / "experiments" / "tbmd_guided_view_intervention"
RECORDS = ROOT / "data" / "cora" / "test" / "eq_by_clip.jsonl"
BASE_CACHE = ROOT / "results" / "eq_renew" / "vanilla_multiclap_eq_embedding_cache"
SOURCE_CACHE = ROOT / "results" / "eq_renew" / "single_model_invariance_embedding_cache"

MODELS = ("laion", "msclap", "m2d", "mga")
MODEL_LABELS = {"laion": "LAION", "msclap": "MS-CLAP", "m2d": "M2D", "mga": "MGA"}
EQ_FORMS = ("key_phrase", "statement", "question", "command", "indirect")
SOURCE = "source_caption"


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def read_records(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern} in {root}; found {len(matches)}")
    return matches[0]


def find_source(model: str, n: int) -> Path:
    matches = []
    for path in sorted(SOURCE_CACHE.glob(f"*text_{model}_*.npz")):
        with np.load(path) as z:
            if SOURCE in z.files and z[SOURCE].shape[0] == n:
                matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source cache for {model}, n={n}; found {len(matches)}")
    return matches[0]


def ranks_local(scores: np.ndarray) -> np.ndarray:
    target = np.diag(scores)
    return (1 + np.sum(scores > target[:, None], axis=1)).astype(np.int32)


def topk_non_target(scores: np.ndarray, k: int = 5) -> np.ndarray:
    n = scores.shape[0]
    kk = min(k, n - 1)
    masked = scores.copy()
    np.fill_diagonal(masked, -np.inf)
    idx = np.argpartition(-masked, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(n)[:, None]
    order = np.argsort(-masked[row, idx], axis=1)
    return idx[row, order]


def tbmd_components(source_scores: np.ndarray, eq_scores: np.ndarray) -> Dict[str, np.ndarray]:
    n = source_scores.shape[0]
    target_src = np.diag(source_scores)
    target_eq = np.diag(eq_scores)
    src_top = topk_non_target(source_scores, k=5)
    eq_top = topk_non_target(eq_scores, k=5)

    tbmd = np.zeros(n, dtype=np.float32)
    tal = (target_src - target_eq).astype(np.float32)
    hnp = np.zeros(n, dtype=np.float32)
    src_margin = np.zeros(n, dtype=np.float32)
    eq_margin = np.zeros(n, dtype=np.float32)

    for i in range(n):
        union = np.unique(np.concatenate([src_top[i], eq_top[i]]))
        src_neg = float(np.max(source_scores[i, union]))
        eq_neg = float(np.max(eq_scores[i, union]))
        src_margin[i] = float(target_src[i] - src_neg)
        eq_margin[i] = float(target_eq[i] - eq_neg)
        hnp[i] = float(eq_neg - src_neg)
        tbmd[i] = float(src_margin[i] - eq_margin[i])

    return {
        "tbmd": tbmd,
        "tal": tal,
        "hnp": hnp,
        "source_margin": src_margin,
        "eq_margin": eq_margin,
    }


def summarize_policy(
    method: str,
    source_rank: np.ndarray,
    eq_rank: np.ndarray,
    tbmd: np.ndarray,
    tal: np.ndarray,
    delta_move: np.ndarray,
    model: str = "all",
    dataset: str = "all",
) -> Dict[str, object]:
    source_hit = source_rank <= 5
    eq_hit = eq_rank <= 5
    return {
        "split": "overall" if model == "all" and dataset == "all" else ("model" if dataset == "all" else "dataset"),
        "model": model,
        "dataset": dataset,
        "method": method,
        "n_units": int(eq_hit.size),
        "source_R@5": float(source_hit.mean()),
        "EQ_R@5": float(eq_hit.mean()),
        "retained_R@5": float(np.logical_and(source_hit, eq_hit).mean()),
        "mean_rankdrop": float(np.mean(eq_rank - source_rank)),
        "mean_TBMD@5": float(np.mean(tbmd)),
        "mean_TAL": float(np.mean(tal)),
        "mean_delta_move": float(np.mean(delta_move)),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}"


def fmt(x: float) -> str:
    return f"{x:.3f}"


def load_model_embeddings(model: str) -> Dict[str, object]:
    text_path = find_one(BASE_CACHE, f"*text_{model}_*.npz")
    audio_path = find_one(BASE_CACHE, f"*audio_{model}_*.npy")
    with np.load(text_path) as z:
        eq_text = {form: l2(z[form]) for form in EQ_FORMS}
    audio = l2(np.load(audio_path))
    n = audio.shape[0]
    with np.load(find_source(model, n)) as z:
        source_text = l2(z[SOURCE])
    return {"source_text": source_text, "eq_text": eq_text, "audio": audio}


def selected_arrays(
    selector: np.ndarray,
    eq_ranks_by_form: Mapping[str, np.ndarray],
    tbmd_by_form: Mapping[str, np.ndarray],
    tal_by_form: Mapping[str, np.ndarray],
    move_by_form: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    forms = list(EQ_FORMS)
    eq_rank_stack = np.stack([eq_ranks_by_form[f] for f in forms], axis=1)
    tbmd_stack = np.stack([tbmd_by_form[f] for f in forms], axis=1)
    tal_stack = np.stack([tal_by_form[f] for f in forms], axis=1)
    move_stack = np.stack([move_by_form[f] for f in forms], axis=1)
    row = np.arange(selector.shape[0])
    return {
        "eq_rank": eq_rank_stack[row, selector],
        "tbmd": tbmd_stack[row, selector],
        "tal": tal_stack[row, selector],
        "move": move_stack[row, selector],
        "selected_form": np.asarray(forms, dtype=object)[selector],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the target-aware TBMD-guided CORA view intervention.")
    parser.add_argument("--eq-jsonl", type=Path, default=RECORDS)
    parser.add_argument("--base-cache-dir", type=Path, default=BASE_CACHE)
    parser.add_argument("--source-cache-dir", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--query-forms", nargs="+", default=list(EQ_FORMS))
    return parser.parse_args()


def main() -> None:
    global OUT, RECORDS, BASE_CACHE, SOURCE_CACHE, MODELS, EQ_FORMS
    args = parse_args()
    OUT = args.output_dir
    RECORDS = args.eq_jsonl
    BASE_CACHE = args.base_cache_dir
    SOURCE_CACHE = args.source_cache_dir
    MODELS = tuple(args.models)
    EQ_FORMS = tuple(args.query_forms)
    OUT.mkdir(parents=True, exist_ok=True)
    records = read_records(RECORDS)
    labels = np.asarray([str(r.get("dataset") or r.get("dataset_slug") or "unknown") for r in records])
    datasets = sorted(set(labels.tolist()))

    overall_buckets: Dict[str, Dict[str, List[np.ndarray]]] = {}
    model_buckets: Dict[tuple[str, str], Dict[str, List[np.ndarray]]] = {}
    dataset_buckets: Dict[tuple[str, str], Dict[str, List[np.ndarray]]] = {}
    selection_count_rows: List[Dict[str, object]] = []

    methods = [
        "All EQ forms",
        "Min delta_move view",
        "Min TAL oracle",
        "Min TBMD@5 oracle",
        "Best-rank oracle",
    ]

    def add(bucket: Dict[str, Dict[str, List[np.ndarray]]], method: str, arrays: Dict[str, np.ndarray]) -> None:
        if method not in bucket:
            bucket[method] = {key: [] for key in arrays}
        for key, value in arrays.items():
            bucket[method][key].append(value)

    for model in MODELS:
        loaded = load_model_embeddings(model)
        source_text = loaded["source_text"]
        eq_text = loaded["eq_text"]
        audio = loaded["audio"]
        source_full = (source_text @ audio.T).astype(np.float32, copy=False)
        eq_full = {form: (eq_text[form] @ audio.T).astype(np.float32, copy=False) for form in EQ_FORMS}
        move_full = {
            form: (1.0 - np.sum(source_text * eq_text[form], axis=1)).astype(np.float32)
            for form in EQ_FORMS
        }

        for dataset in datasets:
            idx = np.flatnonzero(labels == dataset)
            source_local = source_full[np.ix_(idx, idx)]
            source_rank = ranks_local(source_local)

            eq_ranks_by_form: Dict[str, np.ndarray] = {}
            tbmd_by_form: Dict[str, np.ndarray] = {}
            tal_by_form: Dict[str, np.ndarray] = {}
            move_by_form: Dict[str, np.ndarray] = {}

            for form in EQ_FORMS:
                eq_local = eq_full[form][np.ix_(idx, idx)]
                eq_rank = ranks_local(eq_local)
                comp = tbmd_components(source_local, eq_local)
                eq_ranks_by_form[form] = eq_rank
                tbmd_by_form[form] = comp["tbmd"]
                tal_by_form[form] = comp["tal"]
                move_by_form[form] = move_full[form][idx]

            all_eq_rank = np.concatenate([eq_ranks_by_form[f] for f in EQ_FORMS])
            all_source_rank = np.tile(source_rank, len(EQ_FORMS))
            all_tbmd = np.concatenate([tbmd_by_form[f] for f in EQ_FORMS])
            all_tal = np.concatenate([tal_by_form[f] for f in EQ_FORMS])
            all_move = np.concatenate([move_by_form[f] for f in EQ_FORMS])

            method_arrays: Dict[str, Dict[str, np.ndarray]] = {
                "All EQ forms": {
                    "source_rank": all_source_rank,
                    "eq_rank": all_eq_rank,
                    "tbmd": all_tbmd,
                    "tal": all_tal,
                    "move": all_move,
                }
            }

            stacks = {
                "rank": np.stack([eq_ranks_by_form[f] for f in EQ_FORMS], axis=1),
                "tbmd": np.stack([tbmd_by_form[f] for f in EQ_FORMS], axis=1),
                "tal": np.stack([tal_by_form[f] for f in EQ_FORMS], axis=1),
                "move": np.stack([move_by_form[f] for f in EQ_FORMS], axis=1),
            }
            selectors = {
                "Min delta_move view": np.argmin(stacks["move"], axis=1),
                "Min TAL oracle": np.argmin(stacks["tal"], axis=1),
                "Min TBMD@5 oracle": np.argmin(stacks["tbmd"], axis=1),
                "Best-rank oracle": np.argmin(stacks["rank"], axis=1),
            }
            for method, selector in selectors.items():
                sel = selected_arrays(selector, eq_ranks_by_form, tbmd_by_form, tal_by_form, move_by_form)
                method_arrays[method] = {
                    "source_rank": source_rank,
                    "eq_rank": sel["eq_rank"],
                    "tbmd": sel["tbmd"],
                    "tal": sel["tal"],
                    "move": sel["move"],
                }
                unique, counts = np.unique(sel["selected_form"], return_counts=True)
                for form, count in zip(unique, counts):
                    selection_count_rows.append(
                        {
                            "model": MODEL_LABELS[model],
                            "dataset": dataset,
                            "method": method,
                            "query_form": str(form),
                            "count": int(count),
                            "rate": float(count / selector.shape[0]),
                        }
                    )

            for method, arrays in method_arrays.items():
                add(overall_buckets, method, arrays)
                add(model_buckets.setdefault((MODEL_LABELS[model], method), {}), method, arrays)
                add(dataset_buckets.setdefault((dataset, method), {}), method, arrays)

    overall_rows: List[Dict[str, object]] = []
    for method in methods:
        data = overall_buckets[method]
        overall_rows.append(
            summarize_policy(
                method=method,
                source_rank=np.concatenate(data["source_rank"]),
                eq_rank=np.concatenate(data["eq_rank"]),
                tbmd=np.concatenate(data["tbmd"]),
                tal=np.concatenate(data["tal"]),
                delta_move=np.concatenate(data["move"]),
            )
        )

    model_rows: List[Dict[str, object]] = []
    for (model, method), bucket in model_buckets.items():
        data = bucket[method]
        model_rows.append(
            summarize_policy(
                method=method,
                source_rank=np.concatenate(data["source_rank"]),
                eq_rank=np.concatenate(data["eq_rank"]),
                tbmd=np.concatenate(data["tbmd"]),
                tal=np.concatenate(data["tal"]),
                delta_move=np.concatenate(data["move"]),
                model=model,
            )
        )

    dataset_rows: List[Dict[str, object]] = []
    for (dataset, method), bucket in dataset_buckets.items():
        data = bucket[method]
        dataset_rows.append(
            summarize_policy(
                method=method,
                source_rank=np.concatenate(data["source_rank"]),
                eq_rank=np.concatenate(data["eq_rank"]),
                tbmd=np.concatenate(data["tbmd"]),
                tal=np.concatenate(data["tal"]),
                delta_move=np.concatenate(data["move"]),
                dataset=dataset,
            )
        )

    write_csv(OUT / "overall.csv", overall_rows)
    write_csv(OUT / "by_model.csv", sorted(model_rows, key=lambda r: (str(r["model"]), methods.index(str(r["method"])))))
    write_csv(OUT / "by_dataset.csv", sorted(dataset_rows, key=lambda r: (str(r["dataset"]), methods.index(str(r["method"])))))
    write_csv(OUT / "selection_counts.csv", selection_count_rows)

    lines = [
        "# TBMD-Guided EQ View Intervention",
        "",
        "This is a controlled target-aware oracle experiment, not a deployable inference method.",
        "For each source caption, model, and dataset-local candidate pool, the experiment chooses one of the five EQ query forms.",
        "The goal is to test whether TBMD@5 has actionable content: selecting low-TBMD views should reduce target RankDrop better than selecting the text-space-closest view.",
        "",
        "## Overall",
        "",
        "| Method | EQ R@5 | Retained R@5 | Mean RankDrop | Mean TBMD@5 | Mean TAL | Mean delta_move |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['method']} | {pct(float(row['EQ_R@5']))} | {pct(float(row['retained_R@5']))} | "
            f"{fmt(float(row['mean_rankdrop']))} | {fmt(float(row['mean_TBMD@5']))} | "
            f"{fmt(float(row['mean_TAL']))} | {fmt(float(row['mean_delta_move']))} |"
        )

    lines.extend(
        [
            "",
            "## Dataset Breakdown",
            "",
            "| Dataset | Method | EQ R@5 | Retained R@5 | Mean RankDrop | Mean TBMD@5 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(dataset_rows, key=lambda r: (str(r["dataset"]), methods.index(str(r["method"])))):
        lines.append(
            f"| {row['dataset']} | {row['method']} | {pct(float(row['EQ_R@5']))} | "
            f"{pct(float(row['retained_R@5']))} | {fmt(float(row['mean_rankdrop']))} | "
            f"{fmt(float(row['mean_TBMD@5']))} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Min delta_move view` is a target-free text-space control: it picks the EQ form closest to the source caption embedding.",
            "- `Min TAL oracle` is a target-only oracle: it picks the EQ form with the smallest target-alignment loss.",
            "- `Min TBMD@5 oracle` is the proposed diagnostic intervention: it picks the EQ form with the smallest target-boundary margin degradation.",
            "- `Best-rank oracle` is the upper bound from directly selecting the best target rank.",
            "",
            "If `Min TBMD@5 oracle` substantially improves EQ R@5 and reduces Mean RankDrop relative to `Min delta_move view`, then TBMD@5 is not merely descriptive of degradation after the fact; it identifies which equivalent query views are rank-safe in this controlled setting.",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

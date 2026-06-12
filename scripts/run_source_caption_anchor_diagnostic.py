#!/usr/bin/env python3
"""Compute source-caption-anchor embedding diagnostics for the main paper flow."""

from __future__ import annotations

import argparse
import csv
import os
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("CORA_ROOT") or os.environ.get("VAQ_ROOT") or Path(__file__).resolve().parents[1])
MULTICLAP = ROOT
OUT_DIR = ROOT / "experiments" / "source_caption_anchor_diagnostic"

MODELS = ("laion", "msclap", "m2d", "mga")
EQ_FORMS = ("key_phrase", "statement", "question", "command", "indirect")
ANCHOR = "source_caption"

VANILLA_CACHE = Path(os.environ.get("CORA_VANILLA_CACHE", MULTICLAP / "results" / "eq_renew" / "vanilla_multiclap_eq_embedding_cache"))
SOURCE_CACHE = Path(os.environ.get("CORA_SOURCE_CACHE", MULTICLAP / "results" / "eq_renew" / "single_model_invariance_embedding_cache"))
EQ_JSONL = Path(os.environ.get("CORA_EQ_JSONL", MULTICLAP / "data" / "cora" / "test" / "eq_by_clip.jsonl"))


def l2(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern} in {root}; found {len(matches)}")
    return matches[0]


def find_source(model: str, n: int) -> Path:
    matches = []
    for path in sorted(SOURCE_CACHE.glob(f"*text_{model}_*.npz")):
        with np.load(path) as z:
            if ANCHOR in z.files and z[ANCHOR].shape[0] == n:
                matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source_caption cache for {model}, n={n}; found {len(matches)}")
    return matches[0]


def top1_non_target(scores: np.ndarray) -> np.ndarray:
    work = scores.copy()
    n = work.shape[0]
    work[np.arange(n), np.arange(n)] = -np.inf
    return np.argmax(work, axis=1)


def topk_non_target(scores: np.ndarray, k: int = 5) -> np.ndarray:
    work = scores.copy()
    n = work.shape[0]
    work[np.arange(n), np.arange(n)] = -np.inf
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    row = np.arange(n)[:, None]
    order = np.argsort(-work[row, idx], axis=1)
    return idx[row, order]


def ranks(scores: np.ndarray) -> np.ndarray:
    n = scores.shape[0]
    target = scores[np.arange(n), np.arange(n)]
    return 1 + np.sum(scores > target[:, None], axis=1)


def corr(rows: List[Mapping[str, object]], x: str, y: str = "rank_drop") -> float:
    xs = np.asarray([float(r[x]) for r in rows], dtype=np.float64)
    ys = np.asarray([float(r[y]) for r in rows], dtype=np.float64)
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    yr = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def rankdrop_prediction_controls(rows: List[Mapping[str, object]]) -> List[Dict[str, object]]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:  # pragma: no cover
        return [
            {
                "feature_set": "skipped",
                "outcome": "rank_drop",
                "GroupCV_Pearson": float("nan"),
                "GroupCV_Spearman": float("nan"),
                "NumericFeatures": "",
                "N": len(rows),
                "note": str(exc),
            }
        ]

    frame = pd.DataFrame(rows)
    frame["target"] = frame["rank_drop"].astype(float)
    categorical = ["dataset", "model", "query_type"]
    feature_sets = [
        ("fixed_effects_only", []),
        ("query_movement_plus_fe", ["query_move_cosdist"]),
        ("target_delta_plus_fe", ["query_move_cosdist", "target_align_delta"]),
        ("form_hn_margin_plus_fe", ["query_move_cosdist", "embedding_margin_delta"]),
        ("source_hn_margin_plus_fe", ["query_move_cosdist", "source_hn_margin_delta"]),
        ("union_top5_margin_plus_fe", ["query_move_cosdist", "union_top5_margin_delta"]),
        ("directional_components_plus_fe", ["query_move_cosdist", "target_align_delta", "form_hn_align_delta"]),
    ]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    n_splits = int(min(5, frame["audio_id"].nunique()))
    outputs: List[Dict[str, object]] = []
    for name, numeric in feature_sets:
        data = frame[categorical + numeric + ["target", "audio_id"]].replace([np.inf, -np.inf], np.nan).dropna()
        y = data["target"].astype(float)
        pearson_r = float("nan")
        spearman_rho = float("nan")
        if y.nunique() >= 2 and len(data) >= 20 and n_splits >= 2:
            transformers = [("cat", encoder, categorical)]
            if numeric:
                transformers.insert(0, ("num", StandardScaler(), numeric))
            pipeline = Pipeline(
                [
                    ("preprocess", ColumnTransformer(transformers)),
                    ("reg", Ridge(alpha=1.0)),
                ]
            )
            cv = GroupKFold(n_splits=n_splits)
            pred = cross_val_predict(pipeline, data[categorical + numeric], y, groups=data["audio_id"], cv=cv)
            pearson_r = corr([{"pred": p, "target": t} for p, t in zip(pred, y)], "pred", "target")
            spearman_rho = spearman(pred, y.to_numpy())
        outputs.append(
            {
                "feature_set": name,
                "outcome": "rank_drop",
                "GroupCV_Pearson": pearson_r,
                "GroupCV_Spearman": spearman_rho,
                "NumericFeatures": ",".join(numeric),
                "N": int(len(data)),
            }
        )
    return outputs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the source-caption anchored CORA RankDrop/TBMD diagnostic.")
    parser.add_argument("--eq-jsonl", type=Path, default=EQ_JSONL)
    parser.add_argument("--base-cache-dir", type=Path, default=VANILLA_CACHE)
    parser.add_argument("--source-cache-dir", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--query-forms", nargs="+", default=list(EQ_FORMS))
    parser.add_argument("--anchor", default=ANCHOR)
    return parser.parse_args()


def main() -> None:
    global OUT_DIR, VANILLA_CACHE, SOURCE_CACHE, EQ_JSONL, MODELS, EQ_FORMS, ANCHOR
    args = parse_args()
    OUT_DIR = args.output_dir
    VANILLA_CACHE = args.base_cache_dir
    SOURCE_CACHE = args.source_cache_dir
    EQ_JSONL = args.eq_jsonl
    MODELS = tuple(args.models)
    EQ_FORMS = tuple(args.query_forms)
    ANCHOR = args.anchor
    if "statement" not in EQ_FORMS:
        raise ValueError("--query-forms must include statement because cache order validation uses it.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in EQ_JSONL.read_text(encoding="utf-8").splitlines()]
    labels = np.asarray([str(r.get("dataset") or r.get("dataset_slug")) for r in records])
    all_rows: List[Dict[str, object]] = []
    cell_rows: List[Dict[str, object]] = []

    for model in MODELS:
        text_path = find_one(VANILLA_CACHE, f"*text_{model}_*.npz")
        audio_path = find_one(VANILLA_CACHE, f"*audio_{model}_*.npy")
        with np.load(text_path) as z:
            text = {form: l2(z[form]) for form in EQ_FORMS}
        audio = l2(np.load(audio_path))
        source_path = find_source(model, audio.shape[0])
        with np.load(source_path) as z:
            anchor = l2(z[ANCHOR])
            statement_delta = float(np.max(np.abs(z["statement"] - text["statement"])))
            if statement_delta > 1e-5:
                raise ValueError(f"Cache order mismatch for {model}: {statement_delta}")

        scores = {form: (text[form] @ audio.T).astype(np.float32, copy=False) for form in EQ_FORMS}
        scores[ANCHOR] = (anchor @ audio.T).astype(np.float32, copy=False)
        rank = {form: ranks(scores[form]) for form in [ANCHOR, *EQ_FORMS]}
        anchor_target = scores[ANCHOR][np.arange(audio.shape[0]), np.arange(audio.shape[0])]
        anchor_hn = top1_non_target(scores[ANCHOR])
        statement_top5 = topk_non_target(scores[ANCHOR], 5)

        for form in EQ_FORMS:
            form_hn = top1_non_target(scores[form])
            form_top5 = topk_non_target(scores[form], 5)
            form_target = scores[form][np.arange(audio.shape[0]), np.arange(audio.shape[0])]
            form_hn_form = scores[form][np.arange(audio.shape[0]), form_hn]
            form_hn_anchor = scores[ANCHOR][np.arange(audio.shape[0]), form_hn]
            anchor_hn_form = scores[form][np.arange(audio.shape[0]), anchor_hn]
            anchor_hn_anchor = scores[ANCHOR][np.arange(audio.shape[0]), anchor_hn]
            target_delta = form_target - anchor_target
            form_hn_delta = form_hn_form - form_hn_anchor
            margin_delta = (form_target - form_hn_form) - (anchor_target - form_hn_anchor)
            anchor_hn_margin_delta = (form_target - anchor_hn_form) - (anchor_target - anchor_hn_anchor)
            query_move = 1.0 - np.sum(text[form] * anchor, axis=1)

            union_margin_delta = np.empty(audio.shape[0], dtype=np.float32)
            for i in range(audio.shape[0]):
                pool = np.unique(np.concatenate([statement_top5[i], form_top5[i]]))
                anchor_boundary = float(np.max(scores[ANCHOR][i, pool]))
                form_boundary = float(np.max(scores[form][i, pool]))
                union_margin_delta[i] = (form_target[i] - form_boundary) - (anchor_target[i] - anchor_boundary)

            for i in range(audio.shape[0]):
                all_rows.append(
                    {
                        "anchor_design": "source_caption_vs_eq5",
                        "dataset": labels[i],
                        "model": model,
                        "audio_id": records[i].get("audio_id", ""),
                        "anchor_form": ANCHOR,
                        "query_type": form,
                        "anchor_rank": int(rank[ANCHOR][i]),
                        "form_rank": int(rank[form][i]),
                        "rank_drop": int(rank[form][i] - rank[ANCHOR][i]),
                        "anchor_hit@5": bool(rank[ANCHOR][i] <= 5),
                        "form_hit@5": bool(rank[form][i] <= 5),
                        "query_move_cosdist": float(query_move[i]),
                        "target_align_delta": float(target_delta[i]),
                        "form_hn_align_delta": float(form_hn_delta[i]),
                        "hn_minus_target_delta": float(form_hn_delta[i] - target_delta[i]),
                        "embedding_margin_delta": float(margin_delta[i]),
                        "source_hn_margin_delta": float(anchor_hn_margin_delta[i]),
                        "union_top5_margin_delta": float(union_margin_delta[i]),
                    }
                )

            for dataset in sorted(set(labels.tolist())):
                mask = labels == dataset
                cell_rows.append(
                    {
                        "anchor_design": "source_caption_vs_eq5",
                        "dataset": dataset,
                        "model": model,
                        "anchor_form": ANCHOR,
                        "query_type": form,
                        "N": int(mask.sum()),
                        "Anchor_R@5": float(np.mean(rank[ANCHOR][mask] <= 5) * 100),
                        "Form_R@5": float(np.mean(rank[form][mask] <= 5) * 100),
                        "Delta_R@5_vs_anchor": float((np.mean(rank[form][mask] <= 5) - np.mean(rank[ANCHOR][mask] <= 5)) * 100),
                        "MeanRankDrop": float(np.mean(rank[form][mask] - rank[ANCHOR][mask])),
                        "SourceHitMeanRankDrop": float(np.mean((rank[form][mask] - rank[ANCHOR][mask])[rank[ANCHOR][mask] <= 5])) if np.any(rank[ANCHOR][mask] <= 5) else float("nan"),
                        "MeanQueryMoveCosDist": float(np.mean(query_move[mask])),
                        "MeanTargetAlignDelta": float(np.mean(target_delta[mask])),
                        "MeanFormHNMarginDelta": float(np.mean(margin_delta[mask])),
                        "MeanSourceHNMarginDelta": float(np.mean(anchor_hn_margin_delta[mask])),
                        "MeanUnionTop5MarginDelta": float(np.mean(union_margin_delta[mask])),
                    }
                )

    metrics = [
        "query_move_cosdist",
        "target_align_delta",
        "form_hn_align_delta",
        "hn_minus_target_delta",
        "embedding_margin_delta",
        "source_hn_margin_delta",
        "union_top5_margin_delta",
    ]
    corr_rows = [
        {
            "analysis": "source_caption_vs_eq5",
            "outcome": "rank_drop",
            "metric": metric,
            "N": len(all_rows),
            "Pearson": corr(all_rows, metric),
        }
        for metric in metrics
    ]
    n = len(all_rows)
    worsened = sum(1 for r in all_rows if int(r["rank_drop"]) > 0)
    mean_rankdrop = float(np.mean([float(r["rank_drop"]) for r in all_rows]))
    source_hit_rankdrop = float(np.mean([float(r["rank_drop"]) for r in all_rows if r["anchor_hit@5"]]))
    summary = [
        {"metric": "pairwise_rows", "value": n},
        {"metric": "worsened_pairs", "value": worsened},
        {"metric": "worsened_percent", "value": 100.0 * worsened / n},
        {"metric": "mean_rankdrop", "value": mean_rankdrop},
        {"metric": "source_hit_mean_rankdrop", "value": source_hit_rankdrop},
    ]

    write_csv(OUT_DIR / "source_caption_pairwise.csv", all_rows)
    write_csv(OUT_DIR / "source_caption_cells.csv", cell_rows)
    write_csv(OUT_DIR / "source_caption_correlations.csv", corr_rows)
    write_csv(OUT_DIR / "source_caption_summary.csv", summary)
    rankdrop_prediction_rows = rankdrop_prediction_controls(all_rows)
    write_csv(OUT_DIR / "source_caption_rankdrop_prediction_controls.csv", rankdrop_prediction_rows)

    lines = [
        "# Source Caption Anchor Diagnostic",
        "",
        "Anchor: `source_caption`; forms: EQ5.",
        "",
        f"Pairwise rows: `{n}`",
        f"Worsened pairs: `{worsened}` ({100.0 * worsened / n:.3f}%)",
        f"Mean RankDrop: `{mean_rankdrop:.3f}`",
        f"Mean RankDrop among source-caption hits: `{source_hit_rankdrop:.3f}`",
        "",
        "| Metric | Pearson with RankDrop |",
        "|---|---:|",
    ]
    for row in corr_rows:
        lines.append(f"| {row['metric']} | {float(row['Pearson']):.3f} |")
    lines.extend(["", "## RankDrop Prediction Controls", "", "| Feature set | Pearson r | Spearman rho |", "|---|---:|---:|"])
    for row in rankdrop_prediction_rows:
        lines.append(f"| {row['feature_set']} | {float(row['GroupCV_Pearson']):.3f} | {float(row['GroupCV_Spearman']):.3f} |")
    lines.extend(
        [
            "",
            "Files:",
            "- `source_caption_pairwise.csv`",
            "- `source_caption_cells.csv`",
            "- `source_caption_correlations.csv`",
            "- `source_caption_summary.csv`",
            "- `source_caption_rankdrop_prediction_controls.csv`",
        ]
    )
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()

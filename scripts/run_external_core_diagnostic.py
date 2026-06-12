#!/usr/bin/env python3
"""Run external retriever core diagnostics on CORA-local embeddings.

The primary paired outcome is continuous RankDrop rather than binary top-5
boundary crossing. Projection feature sets are evaluated by grouped
cross-validated RankDrop prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EQ_FORMS = ("key_phrase", "statement", "question", "command", "indirect")
ALL_TEXT_FORMS = ("source_caption", *EQ_FORMS)
CAPTION_ADJACENT = {"key_phrase", "statement"}
PRAGMATIC = {"question", "command", "indirect"}


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_name(record: dict) -> str:
    return str(record.get("dataset") or record.get("dataset_slug") or record.get("metadata", {}).get("dataset") or "unknown")


def audio_id(record: dict, index: int) -> str:
    return str(record.get("audio_id") or record.get("clip_id") or record.get("id") or index)


def query_text(record: dict, form: str) -> str:
    if form == "source_caption":
        return str(record.get("source_caption", ""))
    return str(record.get("generated_queries", {}).get(form, ""))


def load_cache(cache_root: Path, model_label: str) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray], dict]:
    cache_dir = cache_root / model_label
    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing cache for {model_label}: {cache_dir}")
    records = load_jsonl(cache_dir / "records.jsonl")
    audio = l2(np.load(cache_dir / "audio_embeddings.npy"))
    raw_text = np.load(cache_dir / "text_embeddings.npz")
    missing = [form for form in ALL_TEXT_FORMS if form not in raw_text.files]
    if missing:
        raise ValueError(f"{cache_dir}/text_embeddings.npz missing forms: {missing}")
    text = {form: l2(raw_text[form]) for form in ALL_TEXT_FORMS}
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if len(records) != len(audio):
        raise ValueError(f"Cache length mismatch for {model_label}: records={len(records)} audio={len(audio)}")
    for form, arr in text.items():
        if len(arr) != len(records):
            raise ValueError(f"Text length mismatch for {model_label}/{form}: {len(arr)} vs {len(records)}")
    return records, audio, text, metadata


def rank_topk(scores: np.ndarray, pool_indices: np.ndarray, target_index: int, top_k: int) -> tuple[int, list[int]]:
    order = np.argsort(-scores, kind="mergesort")
    local_target = int(np.where(pool_indices == target_index)[0][0])
    rank = int(np.where(order == local_target)[0][0] + 1)
    return rank, [int(pool_indices[i]) for i in order[:top_k]]


def best_non_target(indices: list[int], query: np.ndarray, audio: np.ndarray, target_index: int) -> int | None:
    candidates = [idx for idx in indices if idx != target_index]
    if not candidates:
        return None
    scores = audio[candidates] @ query
    return int(candidates[int(np.argmax(scores))])


def margin_delta(q_src: np.ndarray, q_eq: np.ndarray, a_t: np.ndarray, a_h: np.ndarray) -> float:
    src_margin = float(q_src @ a_t - q_src @ a_h)
    eq_margin = float(q_eq @ a_t - q_eq @ a_h)
    return eq_margin - src_margin


def normalized_projection(direction: np.ndarray, basis: np.ndarray) -> float:
    denom = float(np.linalg.norm(basis))
    if denom <= 1e-12:
        return float("nan")
    return float(direction @ basis / denom)


def pearson(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2:
        return float("nan")
    if frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(frame.iloc[:, 0], frame.iloc[:, 1])[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    yr = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def taxonomy(form: str) -> str:
    if form in CAPTION_ADJACENT:
        return "caption_adjacent"
    if form in PRAGMATIC:
        return "pragmatic"
    return "other"


def compute_pairwise(model_label: str, records: list[dict], audio: np.ndarray, text: dict[str, np.ndarray], top_k: int) -> pd.DataFrame:
    datasets: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        datasets.setdefault(dataset_name(record), []).append(i)
    dataset_pools = {name: np.asarray(indices, dtype=np.int64) for name, indices in datasets.items()}

    rows: list[dict] = []
    for i, record in enumerate(records):
        ds = dataset_name(record)
        pool = dataset_pools[ds]
        a_t = audio[i]
        q_src = text["source_caption"][i]
        src_scores = audio[pool] @ q_src
        src_rank, src_top = rank_topk(src_scores, pool, i, top_k)
        source_hit = int(src_rank <= top_k)

        for form in EQ_FORMS:
            q_eq = text[form][i]
            eq_scores = audio[pool] @ q_eq
            eq_rank, eq_top = rank_topk(eq_scores, pool, i, top_k)
            eq_hit = int(eq_rank <= top_k)
            rank_drop = eq_rank - src_rank

            union_hn = best_non_target(sorted(set(src_top).union(eq_top)), q_eq, audio, i)
            form_hn = best_non_target(eq_top, q_eq, audio, i)
            source_hn = best_non_target(src_top, q_src, audio, i)
            fallback_hn = best_non_target([int(pool[j]) for j in np.argsort(-eq_scores)[: max(top_k + 5, 20)]], q_eq, audio, i)
            union_hn = union_hn if union_hn is not None else fallback_hn
            form_hn = form_hn if form_hn is not None else fallback_hn
            source_hn = source_hn if source_hn is not None else fallback_hn
            if union_hn is None:
                continue

            d = q_eq - q_src
            a_union = audio[union_hn]
            target_src = float(q_src @ a_t)
            target_eq = float(q_eq @ a_t)
            hn_src = float(q_src @ a_union)
            hn_eq = float(q_eq @ a_union)
            union_margin_delta = (target_eq - hn_eq) - (target_src - hn_src)

            rows.append(
                {
                    "model": model_label,
                    "dataset": ds,
                    "audio_id": audio_id(record, i),
                    "row_index": i,
                    "query_form": form,
                    "form_group": taxonomy(form),
                    "source_caption": query_text(record, "source_caption"),
                    "eq_query": query_text(record, form),
                    "source_rank": src_rank,
                    "eq_rank": eq_rank,
                    "rank_drop": rank_drop,
                    "source_hit5": source_hit,
                    "eq_hit5": eq_hit,
                    "query_move_cosdist": float(1.0 - q_src @ q_eq),
                    "query_movement_norm": float(np.linalg.norm(d)),
                    "target_align_src": target_src,
                    "target_align_eq": target_eq,
                    "target_align_delta": target_eq - target_src,
                    "union_hn_index": union_hn,
                    "union_hn_audio_id": audio_id(records[union_hn], union_hn),
                    "target_hn_margin_src": target_src - hn_src,
                    "target_hn_margin_eq": target_eq - hn_eq,
                    "union_top5_margin_delta": union_margin_delta,
                    "tbmd@5": -union_margin_delta,
                    "embedding_margin_delta": union_margin_delta,
                    "hn_minus_target_delta": (hn_eq - target_eq) - (hn_src - target_src),
                    "form_hn_margin_delta": margin_delta(q_src, q_eq, a_t, audio[form_hn]) if form_hn is not None else float("nan"),
                    "source_hn_margin_delta": margin_delta(q_src, q_eq, a_t, audio[source_hn]) if source_hn is not None else float("nan"),
                    "p_audio": normalized_projection(d, a_t - q_src),
                    "p_boundary": normalized_projection(d, a_t - a_union),
                }
            )
    return pd.DataFrame(rows)


def source_hit_mean(group: pd.DataFrame, col: str) -> float:
    source_hits = group[group["source_hit5"] == 1]
    return float(source_hits[col].mean()) if len(source_hits) else float("nan")


def summarize_model(df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in df.groupby("model", sort=False):
        pred_match = pred_df[(pred_df["model"] == model) & (pred_df["feature_set"] == "full_projection") & (pred_df["form_group"] == "all")]
        rows.append(
            {
                "model": model,
                "n_pairs": len(group),
                "source_caption_r5": group["source_hit5"].mean(),
                "eq_r5": group["eq_hit5"].mean(),
                "mean_rankdrop": group["rank_drop"].mean(),
                "source_hit_mean_rankdrop": source_hit_mean(group, "rank_drop"),
                "corr_move_rankdrop": pearson(group["query_move_cosdist"], group["rank_drop"]),
                "corr_target_delta_rankdrop": pearson(group["target_align_delta"], group["rank_drop"]),
                "corr_tbmd_rankdrop": pearson(group["tbmd@5"], group["rank_drop"]),
                "full_projection_pearson": float(pred_match["pearson"].iloc[0]) if not pred_match.empty else float("nan"),
                "full_projection_spearman": float(pred_match["spearman"].iloc[0]) if not pred_match.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_taxonomy(df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, form_group), group in df.groupby(["model", "form_group"], sort=False):
        pred_match = pred_df[(pred_df["model"] == model) & (pred_df["form_group"] == form_group) & (pred_df["feature_set"] == "full_projection")]
        rows.append(
            {
                "model": model,
                "form_group": form_group,
                "n_pairs": len(group),
                "eq_r5": group["eq_hit5"].mean(),
                "mean_rankdrop": group["rank_drop"].mean(),
                "source_hit_mean_rankdrop": source_hit_mean(group, "rank_drop"),
                "mean_tbmd@5": group["tbmd@5"].mean(),
                "mean_target_align_delta": group["target_align_delta"].mean(),
                "mean_p_audio": group["p_audio"].mean(),
                "mean_p_boundary": group["p_boundary"].mean(),
                "projection_spearman": float(pred_match["spearman"].iloc[0]) if not pred_match.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def cv_rankdrop(frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> tuple[float, float, str]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:  # pragma: no cover
        return float("nan"), float("nan"), f"skipped: {exc}"

    cols = numeric_features + categorical_features
    data = frame[cols + ["rank_drop", "audio_id"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        return float("nan"), float("nan"), "skipped: fewer than 20 rows"
    y = data["rank_drop"].astype(float)
    if y.nunique() < 2:
        return float("nan"), float("nan"), "skipped: one target value"
    n_splits = int(min(5, data["audio_id"].nunique()))
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))
    pipeline = Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers)),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    cv = GroupKFold(n_splits=n_splits)
    try:
        pred = cross_val_predict(pipeline, data[cols], y, groups=data["audio_id"], cv=cv)
        return float(np.corrcoef(pred, y)[0, 1]), spearman(pred, y.to_numpy()), f"{n_splits}-fold grouped CV by audio_id"
    except Exception as exc:  # pragma: no cover
        return float("nan"), float("nan"), f"skipped: {exc}"


def compute_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    feature_sets = {
        "fixed_effects": ([], ["dataset", "query_form"]),
        "query_movement": (["query_move_cosdist", "query_movement_norm"], ["dataset", "query_form"]),
        "audio_projection": (["p_audio", "query_movement_norm"], ["dataset", "query_form"]),
        "boundary_projection": (["p_boundary", "query_movement_norm"], ["dataset", "query_form"]),
        "full_projection": (["p_audio", "p_boundary", "query_move_cosdist", "query_movement_norm"], ["dataset", "query_form"]),
    }
    rows = []
    for model, group in df.groupby("model", sort=False):
        for feature_set, (numeric, categorical) in feature_sets.items():
            pearson_r, spearman_rho, note = cv_rankdrop(group, numeric, categorical)
            rows.append({"model": model, "form_group": "all", "feature_set": feature_set, "pearson": pearson_r, "spearman": spearman_rho, "note": note})
        for form_group, sub in group.groupby("form_group", sort=False):
            for feature_set, (numeric, categorical) in feature_sets.items():
                pearson_r, spearman_rho, note = cv_rankdrop(sub, numeric, categorical)
                rows.append({"model": model, "form_group": form_group, "feature_set": feature_set, "pearson": pearson_r, "spearman": spearman_rho, "note": note})
    return pd.DataFrame(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, summary: pd.DataFrame, taxonomy_summary: pd.DataFrame, pred_df: pd.DataFrame, pairwise_rows: int) -> None:
    pred_wide = pred_df[pred_df["form_group"] == "all"].pivot(index="model", columns="feature_set", values="spearman").reset_index()
    report = [
        "# External Retriever Core Diagnostic Report",
        "",
        "This report is generated from CORA-local embedding caches. It does not copy embeddings from the external repositories.",
        "",
        f"Pairwise rows: `{pairwise_rows}`",
        "",
        "## Model Summary",
        "",
        markdown_table(
            summary,
            [
                "model",
                "eq_r5",
                "mean_rankdrop",
                "source_hit_mean_rankdrop",
                "corr_move_rankdrop",
                "corr_target_delta_rankdrop",
                "corr_tbmd_rankdrop",
                "full_projection_spearman",
            ],
        ),
        "",
        "## Form Taxonomy Summary",
        "",
        markdown_table(
            taxonomy_summary,
            [
                "model",
                "form_group",
                "eq_r5",
                "mean_rankdrop",
                "source_hit_mean_rankdrop",
                "mean_tbmd@5",
                "mean_p_boundary",
                "projection_spearman",
            ],
        ),
        "",
        "## Projection Spearman Feature Sets",
        "",
        markdown_table(pred_wide, [col for col in ["model", "fixed_effects", "query_movement", "audio_projection", "boundary_projection", "full_projection"] if col in pred_wide.columns]),
        "",
        "## Interpretation Contract",
        "",
        "Use these outputs to test whether target/boundary directional features explain paired EQ RankDrop better than raw query movement in RobustCLAP/OEA. Cross-model cosine magnitudes should not be compared directly; compare within-model rank degradation, correlations, and grouped prediction patterns.",
        "",
        "RobustCLAP numbers use the local embedding cache generated with `630k-audioset-best.pt`, `HTSAT-tiny`, `tmodel=roberta`, and an explicit RoBERTa tokenizer override.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Action 1 diagnostics from external retriever embedding caches.")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "results/eq_renew/external_retriever_embedding_cache")
    parser.add_argument("--models", nargs="+", default=["oea_qwen3b_cl", "oea_qwen3b_ac", "robustclap"], help="Cache labels to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments/external_retriever_core_diagnostic")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_frames = []
    metadata = []
    for model_label in args.models:
        records, audio, text, meta = load_cache(args.cache_root, model_label)
        all_frames.append(compute_pairwise(model_label, records, audio, text, args.top_k))
        metadata.append({"model_label": model_label, **meta})

    pairwise = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if pairwise.empty:
        raise SystemExit("No pairwise rows produced.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = compute_prediction_rows(pairwise)
    summary = summarize_model(pairwise, pred_df)
    taxonomy_summary = summarize_taxonomy(pairwise, pred_df)

    pairwise.to_csv(args.output_dir / "external_pairwise.csv", index=False)
    summary.to_csv(args.output_dir / "external_model_summary.csv", index=False)
    taxonomy_summary.to_csv(args.output_dir / "external_form_taxonomy_summary.csv", index=False)
    pred_df.to_csv(args.output_dir / "external_projection_rankdrop_prediction.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({"models": metadata, "top_k": args.top_k}, indent=2), encoding="utf-8")
    write_report(args.output_dir, summary, taxonomy_summary, pred_df, len(pairwise))
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

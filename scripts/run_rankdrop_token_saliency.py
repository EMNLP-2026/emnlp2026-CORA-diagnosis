#!/usr/bin/env python3
"""RankDrop-based boundary-margin token saliency diagnostic.

This reruns the Appendix-J style token saliency audit after replacing the
HitDrop outcome with continuous RankDrop. The unchanged CORA query is used;
token-level Gradient x Input saliency is computed for the target-vs-local-HN
boundary margin, and the resulting token-group shares are summarized against
RankDrop groups and continuous RankDrop prediction.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE_MODELS = ("laion", "msclap", "m2d", "mga")
DEFAULT_QUERY_FORMS = ("statement", "question", "command", "indirect")
EQ_FORMS = ("key_phrase", "statement", "question", "command", "indirect")
OUT_DIR = ROOT / "experiments" / "rankdrop_token_saliency"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OLD = load_module("boundary_margin_token_saliency_old", Path(__file__).with_name("token_saliency_helpers.py"))
CORE = load_module("external_core_diagnostic", Path(__file__).with_name("run_external_core_diagnostic.py"))

# Patch the helper module globals so model config paths and cache defaults
# resolve against this repository root.
OLD.PROJECT_ROOT = ROOT
OLD.CONFIG_PATH = ROOT / "config.yaml"
OLD.BASE_CACHE = ROOT / "results/eq_renew/vanilla_multiclap_eq_embedding_cache"
OLD.SOURCE_CACHE = ROOT / "results/eq_renew/single_model_invariance_embedding_cache"
OLD.EQ_JSONL = ROOT / "data/cora/test/eq_by_clip.jsonl"
CORE.PROJECT_ROOT = ROOT


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if len(xx) < 2 or float(np.std(xx)) == 0.0 or float(np.std(yy)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xx = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    yy = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    return pearson(xx, yy)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    view = df[columns].head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def rank_group(rank_drop: float) -> str:
    return "rank_worsened" if float(rank_drop) > 0 else "rank_improved_or_stable"


def dataset_pools(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    pools: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        pools.setdefault(CORE.dataset_name(record), []).append(i)
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in pools.items()}


def strongest_form_hn(row: pd.Series, records: Sequence[Mapping[str, object]], audio: np.ndarray, text: dict[str, np.ndarray], pools: dict[str, np.ndarray]) -> int:
    """Highest-scoring non-target audio under the CORA query."""
    i = int(row["row_index"])
    form = str(row["query_form"])
    pool = pools[str(row["dataset"])]
    q = text[form][i]
    scores = audio[pool] @ q
    target_pos = np.where(pool == i)[0]
    if len(target_pos):
        scores[int(target_pos[0])] = -np.inf
    hn = int(pool[int(np.argmax(scores))])
    return hn


def sample_rankdrop_balanced(pairwise: pd.DataFrame, query_forms: list[str], max_per_cell: int, seed: int, source_hit_only: bool) -> pd.DataFrame:
    eligible = pairwise[pairwise["query_form"].isin(query_forms)].copy()
    if source_hit_only:
        eligible = eligible[eligible["source_hit5"] == 1].copy()
    eligible["rank_group"] = eligible["rank_drop"].map(rank_group)
    eligible["rank_worsened"] = (eligible["rank_drop"] > 0).astype(int)
    if max_per_cell <= 0:
        return eligible.reset_index(drop=True)
    pieces = []
    for _, group in eligible.groupby(["dataset", "query_form", "rank_group"], sort=False):
        n = min(max_per_cell, len(group))
        pieces.append(group.sample(n=n, random_state=seed) if len(group) > n else group)
    return pd.concat(pieces, ignore_index=True) if pieces else eligible.iloc[0:0].reset_index(drop=True)


def pair_identity(row: pd.Series) -> dict[str, Any]:
    keep = [
        "model",
        "dataset",
        "audio_id",
        "row_index",
        "query_form",
        "form_group",
        "source_hit5",
        "eq_hit5",
        "source_rank",
        "eq_rank",
        "rank_drop",
        "rank_group",
        "rank_worsened",
        "query_move_cosdist",
        "query_movement_norm",
        "target_align_delta",
        "target_hn_margin_eq",
        "union_top5_margin_delta",
        "tbmd@5",
        "p_audio",
        "p_boundary",
    ]
    out = {key: row[key] for key in keep if key in row.index}
    if "target_align_delta" in out:
        out["tal"] = -float(out["target_align_delta"])
    if "tbmd@5" in out:
        out["tbmd"] = float(out["tbmd@5"])
    elif "union_top5_margin_delta" in out:
        out["tbmd"] = -float(out["union_top5_margin_delta"])
    return out


def run_model(model_label: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if model_label not in OLD.SUPPORTED_GRADIENT_MODELS:
        return pd.DataFrame(), pd.DataFrame(), {"model": model_label, "status": "skipped", "note": "gradient adapter unavailable"}

    records, audio, text, metadata = OLD.load_base_cache(CORE, args, model_label)
    pools = dataset_pools(records)
    pairwise = CORE.compute_pairwise(model_label, records, audio, text, args.top_k)
    selected = sample_rankdrop_balanced(pairwise, args.query_forms, args.max_per_cell, args.seed, args.source_hit_only)
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame(), {"model": model_label, "status": "skipped", "note": "no eligible rank-drop pairs"}

    runner = OLD.BoundaryMarginSaliencyRunner(model_label, args)
    pair_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    failures = 0

    for pair_num, (_, row) in enumerate(selected.iterrows(), start=1):
        record_index = int(row["row_index"])
        form = str(row["query_form"])
        hn_index = strongest_form_hn(row, records, audio, text, pools)
        source_caption = CORE.query_text(records[record_index], "source_caption")
        eq_query = CORE.query_text(records[record_index], form)
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
                "local_hn_index": hn_index,
                "local_hn_audio_id": CORE.audio_id(records[hn_index], hn_index),
                "saliency_target_score": sal.target_score,
                "saliency_hard_negative_score": sal.hard_negative_score,
                "saliency_boundary_margin": sal.boundary_margin,
                "token_count": len(sal.tokens),
            }
        )
        base.update(OLD.group_shares(sal.tokens))
        pair_rows.append(base)
        for token in sal.tokens:
            token_rows.append({**base, **token})
        if args.progress_every and pair_num % args.progress_every == 0:
            print(f"[{model_label}] rankdrop saliency pairs {pair_num}/{len(selected)}", flush=True)

    status = {
        "model": model_label,
        "status": "completed",
        "selected_pairs": int(len(selected)),
        "completed_pairs": int(len(pair_rows)),
        "failed_pairs": int(failures),
        "metadata": metadata,
    }
    return pd.DataFrame(pair_rows), pd.DataFrame(token_rows), status


def summarize_rank_groups(pair_detail: pd.DataFrame) -> pd.DataFrame:
    if pair_detail.empty:
        return pd.DataFrame()
    cols = [
        "acoustic_content_positive_share",
        "acoustic_content_negative_share",
        "function_form_negative_share",
        "query_intent_negative_share",
        "function_form_positive_share",
        "query_intent_positive_share",
        "saliency_boundary_margin",
        "token_count",
        "rank_drop",
    ]
    grouped = pair_detail.groupby(["model", "rank_group"], sort=False)
    return grouped.agg(n_pairs=("pair_id", "nunique"), **{col: (col, "mean") for col in cols if col in pair_detail.columns}).reset_index()


def summarize_by_form(pair_detail: pd.DataFrame) -> pd.DataFrame:
    if pair_detail.empty:
        return pd.DataFrame()
    cols = [
        "rank_drop",
        "acoustic_content_positive_share",
        "acoustic_content_negative_share",
        "function_form_negative_share",
        "query_intent_negative_share",
        "saliency_boundary_margin",
    ]
    grouped = pair_detail.groupby(["model", "query_form", "rank_group"], sort=False)
    return grouped.agg(n_pairs=("pair_id", "nunique"), **{col: (col, "mean") for col in cols if col in pair_detail.columns}).reset_index()


def correlation_summary(pair_detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "saliency_boundary_margin",
        "acoustic_content_positive_share",
        "acoustic_content_negative_share",
        "function_form_negative_share",
        "query_intent_negative_share",
        "function_form_positive_share",
        "query_intent_positive_share",
        "token_count",
    ]
    rows = []
    for model, group in pair_detail.groupby("model", sort=False):
        for metric in metrics:
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_pairs": int(group[["rank_drop", metric]].dropna().shape[0]),
                    "pearson": pearson(group[metric], group["rank_drop"]),
                    "spearman": spearman(group[metric], group["rank_drop"]),
                }
            )
    for metric in metrics:
        rows.append(
            {
                "model": "pooled",
                "metric": metric,
                "n_pairs": int(pair_detail[["rank_drop", metric]].dropna().shape[0]),
                "pearson": pearson(pair_detail[metric], pair_detail["rank_drop"]),
                "spearman": spearman(pair_detail[metric], pair_detail["rank_drop"]),
            }
        )
    return pd.DataFrame(rows)


def rankdrop_prediction(pair_detail: pd.DataFrame) -> pd.DataFrame:
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        return pd.DataFrame([{"model": "pooled", "feature_set": "skipped", "note": str(exc)}])

    token_features = [
        "acoustic_content_positive_share",
        "acoustic_content_negative_share",
        "function_form_negative_share",
        "query_intent_negative_share",
        "function_form_positive_share",
        "query_intent_positive_share",
    ]
    feature_sets = [
        ("fixed_effects_only", []),
        ("plus_query_movement", ["query_move_cosdist", "query_movement_norm"]),
        ("plus_token_group_saliency", token_features),
        ("plus_boundary_features", ["query_move_cosdist", "query_movement_norm", "tal", "tbmd", "p_boundary"]),
        ("boundary_plus_token_saliency", ["query_move_cosdist", "query_movement_norm", "tal", "tbmd", "p_boundary", *token_features]),
    ]

    rows = []
    frames = [(model, group.copy()) for model, group in pair_detail.groupby("model", sort=False)]
    frames.append(("pooled", pair_detail.copy()))
    for model, frame in frames:
        for name, numeric_features in feature_sets:
            cols = ["rank_drop", "audio_id", "model", "dataset", "query_form", *numeric_features]
            data = frame[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["rank_drop", *numeric_features]).copy()
            if len(data) < 20 or data["rank_drop"].std() == 0 or data["audio_id"].nunique() < 2:
                rows.append({"model": model, "feature_set": name, "n_pairs": len(data), "pearson": float("nan"), "spearman": float("nan"), "note": "skipped"})
                continue
            x_dicts = []
            for _, row in data.iterrows():
                item: dict[str, object] = {
                    f"model={row['model']}": 1.0,
                    f"dataset={row['dataset']}": 1.0,
                    f"query_form={row['query_form']}": 1.0,
                }
                for feature in numeric_features:
                    item[feature] = float(row[feature])
                x_dicts.append(item)
            y = data["rank_drop"].astype(float).to_numpy()
            n_splits = min(5, int(data["audio_id"].nunique()))
            cv = GroupKFold(n_splits=n_splits)
            reg = make_pipeline(DictVectorizer(sparse=True), StandardScaler(with_mean=False), Ridge(alpha=1.0, solver="lsqr"))
            pred = cross_val_predict(reg, x_dicts, y, groups=data["audio_id"].astype(str).to_numpy(), cv=cv)
            rows.append(
                {
                    "model": model,
                    "feature_set": name,
                    "n_pairs": len(data),
                    "pearson": pearson(pred, y),
                    "spearman": spearman(pred, y),
                    "note": f"{n_splits}-fold GroupKFold by audio_id",
                }
            )
    return pd.DataFrame(rows)


def permutation_sanity(token_detail: pd.DataFrame, seed: int) -> pd.DataFrame:
    if token_detail.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    real_pairs = []
    perm_pairs = []
    for pair_id, group in token_detail.groupby("pair_id", sort=False):
        head = group.iloc[0][["model", "dataset", "audio_id", "query_form", "rank_group", "rank_drop"]].to_dict()
        real_pairs.append({"pair_id": pair_id, **head, **OLD.group_shares(group.to_dict("records"))})
        records = group.to_dict("records")
        labels = [str(row.get("token_group", "other")) for row in records]
        rng.shuffle(labels)
        permuted = []
        for row, label in zip(records, labels):
            item = dict(row)
            item["token_group"] = label
            permuted.append(item)
        perm_pairs.append({"pair_id": pair_id, **head, **OLD.group_shares(permuted)})
    real = pd.DataFrame(real_pairs)
    perm = pd.DataFrame(perm_pairs)
    metrics = [
        "acoustic_content_positive_share",
        "acoustic_content_negative_share",
        "function_form_negative_share",
        "query_intent_negative_share",
    ]
    rows = []
    for (model, group_name), real_group in real.groupby(["model", "rank_group"], sort=False):
        perm_group = perm[(perm["model"] == model) & (perm["rank_group"] == group_name)]
        for metric in metrics:
            rows.append(
                {
                    "model": model,
                    "rank_group": group_name,
                    "metric": metric,
                    "real_mean": float(real_group[metric].mean()),
                    "permuted_mean": float(perm_group[metric].mean()),
                    "real_minus_permuted": float(real_group[metric].mean() - perm_group[metric].mean()),
                    "n_pairs": int(len(real_group)),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    pair_detail: pd.DataFrame,
    group_summary: pd.DataFrame,
    form_summary: pd.DataFrame,
    corr_summary: pd.DataFrame,
    prediction: pd.DataFrame,
    sanity: pd.DataFrame,
    status: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    top_corr = corr_summary[corr_summary["model"].eq("pooled")].copy()
    top_corr = top_corr.sort_values("spearman", key=lambda s: s.abs(), ascending=False)
    report = [
        "# RankDrop Token Saliency Report",
        "",
        "This reruns the Appendix-J token saliency audit with RankDrop rather than HitDrop as the outcome.",
        "The CORA query is left unchanged. We compute token-level Gradient x Input saliency for the local target-vs-hard-negative boundary margin.",
        "",
        f"Models requested: `{', '.join(args.models)}`",
        f"Query forms: `{', '.join(args.query_forms)}`",
        f"Sampling: max `{args.max_per_cell}` per model x dataset x query_form x RankDrop group.",
        f"Source-hit only: `{args.source_hit_only}`",
        f"Pair rows: `{len(pair_detail)}`",
        "",
        "## Run Status",
        "",
        markdown_table(status, [col for col in ["model", "status", "selected_pairs", "completed_pairs", "failed_pairs", "note"] if col in status.columns], max_rows=40),
        "",
        "## Main Table 1: Token-Group Attribution by RankDrop Group",
        "",
        markdown_table(
            group_summary,
            [
                "model",
                "rank_group",
                "n_pairs",
                "rank_drop",
                "acoustic_content_positive_share",
                "acoustic_content_negative_share",
                "function_form_negative_share",
                "query_intent_negative_share",
                "saliency_boundary_margin",
            ],
            max_rows=80,
        ),
        "",
        "## Main Table 2: Token-Saliency Correlation with RankDrop",
        "",
        markdown_table(top_corr, ["model", "metric", "n_pairs", "pearson", "spearman"], max_rows=40),
        "",
        "## Main Table 3: Grouped-CV RankDrop Prediction",
        "",
        markdown_table(prediction, ["model", "feature_set", "n_pairs", "pearson", "spearman", "note"], max_rows=120),
        "",
        "## Form-Level Check",
        "",
        markdown_table(
            form_summary,
            [
                "model",
                "query_form",
                "rank_group",
                "n_pairs",
                "rank_drop",
                "acoustic_content_positive_share",
                "function_form_negative_share",
                "query_intent_negative_share",
            ],
            max_rows=120,
        ),
        "",
        "## Random Token-Group Label Sanity Check",
        "",
        markdown_table(sanity, ["model", "rank_group", "metric", "real_mean", "permuted_mean", "real_minus_permuted", "n_pairs"], max_rows=120),
        "",
        "## Interpretation Contract",
        "",
        "Positive token saliency means the token locally supports the marked target over the selected local hard negative. Negative token saliency means the token locally supports the hard negative over the target. Group shares are normalized within each unchanged query by total absolute token saliency. These are local gradient attributions of the boundary margin, not causal token-deletion evidence.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RankDrop-based boundary-margin token saliency diagnostics.")
    parser.add_argument("--base-cache-dir", type=Path, default=ROOT / "results/eq_renew/vanilla_multiclap_eq_embedding_cache")
    parser.add_argument("--source-cache-dir", type=Path, default=ROOT / "results/eq_renew/single_model_invariance_embedding_cache")
    parser.add_argument("--eq-jsonl", type=Path, default=ROOT / "data/cora/test/eq_by_clip.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--models", nargs="+", default=list(BASE_MODELS))
    parser.add_argument("--query-forms", nargs="+", choices=list(EQ_FORMS), default=list(DEFAULT_QUERY_FORMS))
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-per-cell", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-hit-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_frames = []
    token_frames = []
    statuses = []
    for model in args.models:
        print(f"[rankdrop-saliency] starting {model}", flush=True)
        pair_detail, token_detail, status = run_model(model, args)
        pair_frames.append(pair_detail)
        token_frames.append(token_detail)
        statuses.append(status)
        print(f"[rankdrop-saliency] {model}: {status}", flush=True)

    pair_all = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    token_all = pd.concat(token_frames, ignore_index=True) if token_frames else pd.DataFrame()
    status_df = pd.DataFrame(statuses)
    status_df.to_csv(args.output_dir / "rankdrop_token_saliency_status.csv", index=False)
    if pair_all.empty:
        raise SystemExit("No RankDrop token saliency rows produced.")

    group_summary = summarize_rank_groups(pair_all)
    form_summary = summarize_by_form(pair_all)
    corr_summary = correlation_summary(pair_all)
    prediction = rankdrop_prediction(pair_all)
    sanity = permutation_sanity(token_all, args.seed)

    pair_all.to_csv(args.output_dir / "rankdrop_token_saliency_pairs.csv", index=False)
    token_all.to_csv(args.output_dir / "rankdrop_token_saliency_tokens.csv", index=False)
    group_summary.to_csv(args.output_dir / "rankdrop_token_saliency_group_summary.csv", index=False)
    form_summary.to_csv(args.output_dir / "rankdrop_token_saliency_form_summary.csv", index=False)
    corr_summary.to_csv(args.output_dir / "rankdrop_token_saliency_correlations.csv", index=False)
    prediction.to_csv(args.output_dir / "rankdrop_token_saliency_prediction.csv", index=False)
    sanity.to_csv(args.output_dir / "rankdrop_token_saliency_permutation_sanity.csv", index=False)
    write_report(args.output_dir, pair_all, group_summary, form_summary, corr_summary, prediction, sanity, status_df, args)
    print(f"Wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

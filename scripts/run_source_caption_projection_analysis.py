#!/usr/bin/env python3
"""Source-caption anchored projection analysis for the main paper flow.

This reruns the projection analysis with the same anchor design as Main 2:
source_caption -> EQ5. The primary outcome is continuous RankDrop rather than
binary top-5 boundary crossing.
"""

from __future__ import annotations

import argparse
import csv
import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get('CORA_ROOT') or os.environ.get('VAQ_ROOT') or Path(__file__).resolve().parents[1])
MULTICLAP = ROOT
OUT_DIR = ROOT / 'experiments' / 'source_caption_projection_analysis'

MODELS = ('laion', 'msclap', 'm2d', 'mga')
EQ_FORMS = ('key_phrase', 'statement', 'question', 'command', 'indirect')
ANCHOR = 'source_caption'
STATEMENT = 'statement'

VANILLA_CACHE = Path(os.environ.get('CORA_VANILLA_CACHE', MULTICLAP / 'results' / 'eq_renew' / 'vanilla_multiclap_eq_embedding_cache'))
SOURCE_CACHE = Path(os.environ.get('CORA_SOURCE_CACHE', MULTICLAP / 'results' / 'eq_renew' / 'single_model_invariance_embedding_cache'))
EQ_JSONL = Path(os.environ.get('CORA_EQ_JSONL', MULTICLAP / 'data' / 'cora' / 'test' / 'eq_by_clip.jsonl'))


def l2(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def tangent_at(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        xx = x[None, :]
        return xx - np.sum(q * xx, axis=1, keepdims=True) * q
    return x - np.sum(q * x, axis=1, keepdims=True) * q


def row_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(np.asarray(x, dtype=np.float64)).rank(method='average').to_numpy()
    yr = pd.Series(np.asarray(y, dtype=np.float64)).rank(method='average').to_numpy()
    return corr(xr, yr)


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {pattern} in {root}; found {len(matches)}')
    return matches[0]


def find_source_caption(model: str, n: int) -> Path:
    matches: List[Path] = []
    for path in sorted(SOURCE_CACHE.glob(f'*text_{model}_*.npz')):
        with np.load(path) as z:
            if ANCHOR in z.files and z[ANCHOR].shape[0] == n:
                matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one source_caption cache for {model}, n={n}; found {len(matches)}')
    return matches[0]


def topk_non_target(scores: np.ndarray, k: int = 5) -> np.ndarray:
    work = scores.copy()
    n = min(work.shape)
    work[np.arange(n), np.arange(n)] = -np.inf
    k = min(k, work.shape[1] - 1)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    row = np.arange(work.shape[0])[:, None]
    order = np.argsort(-work[row, idx], axis=1)
    return idx[row, order]


def ranks(scores: np.ndarray) -> np.ndarray:
    n = scores.shape[0]
    target_scores = scores[np.arange(n), np.arange(n)]
    return 1 + np.sum(scores > target_scores[:, None], axis=1)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_labels(records: Iterable[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([str(r.get('dataset') or r.get('dataset_slug') or 'unknown') for r in records])


def load_model(model: str):
    text_path = find_one(VANILLA_CACHE, f'*text_{model}_*.npz')
    audio_path = find_one(VANILLA_CACHE, f'*audio_{model}_*.npy')
    with np.load(text_path) as z:
        text = {form: l2(z[form]) for form in EQ_FORMS}
    audio = l2(np.load(audio_path))
    source_path = find_source_caption(model, audio.shape[0])
    with np.load(source_path) as z:
        anchor = l2(z[ANCHOR])
        statement_delta = float(np.max(np.abs(z[STATEMENT] - text[STATEMENT])))
        if statement_delta > 1e-5:
            raise ValueError(f'Cache order mismatch for {model}: statement max diff={statement_delta}')
    text[ANCHOR] = anchor
    return text, audio


def projection_rows_for_model(model: str, records: Sequence[Mapping[str, object]]):
    labels = dataset_labels(records)
    text, audio = load_model(model)
    anchor = text[ANCHOR]
    statement = text[STATEMENT]
    scores = {form: (text[form] @ audio.T).astype(np.float32, copy=False) for form in (ANCHOR, *EQ_FORMS)}
    rank = {form: ranks(scores[form]) for form in (ANCHOR, *EQ_FORMS)}
    anchor_top5 = topk_non_target(scores[ANCHOR], 5)

    g_audio = l2(tangent_at(anchor, audio))
    g_statement = l2(tangent_at(anchor, statement - anchor))

    rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    centroid_rows: List[Dict[str, object]] = []

    for form in EQ_FORMS:
        q_form = text[form]
        delta = q_form - anchor
        centroid = (q_form.mean(axis=0) - anchor.mean(axis=0)).astype(np.float32)
        residual = delta - centroid[None, :]
        r_shift = tangent_at(anchor, delta)
        r_centroid = tangent_at(anchor, centroid)
        r_residual = tangent_at(anchor, residual)

        form_top5 = topk_non_target(scores[form], 5)
        hard_idx = np.empty(audio.shape[0], dtype=np.int64)
        for i in range(audio.shape[0]):
            pool = np.unique(np.concatenate([anchor_top5[i], form_top5[i]]))
            hard_idx[i] = pool[np.argmax(scores[ANCHOR][i, pool])]
        hard_neg = audio[hard_idx]
        g_boundary = l2(tangent_at(anchor, audio - hard_neg))

        p_audio = row_dot(r_shift, g_audio)
        p_boundary = row_dot(r_shift, g_boundary)
        p_statement = row_dot(r_shift, g_statement)
        p_centroid_boundary = row_dot(r_centroid, g_boundary)
        p_residual_boundary = row_dot(r_residual, g_boundary)
        move_norm = np.linalg.norm(r_shift, axis=1)

        rank_anchor = rank[ANCHOR]
        rank_form = rank[form]
        rank_drop = rank_form - rank_anchor
        rank_worsened = rank_drop > 0
        rank_stable_or_improved = ~rank_worsened

        for i in range(audio.shape[0]):
            rows.append(
                {
                    'anchor_design': 'source_caption_vs_eq5',
                    'model': model,
                    'dataset': labels[i],
                    'audio_id': records[i].get('audio_id', ''),
                    'form': form,
                    'rank_anchor': int(rank_anchor[i]),
                    'rank_form': int(rank_form[i]),
                    'rank_drop': int(rank_drop[i]),
                    'anchor_hit5': int(rank_anchor[i] <= 5),
                    'form_hit5': int(rank_form[i] <= 5),
                    'rank_worsened': int(rank_worsened[i]),
                    'query_move_norm': float(move_norm[i]),
                    'p_audio': float(p_audio[i]),
                    'p_boundary': float(p_boundary[i]),
                    'p_statement': float(p_statement[i]),
                    'audio_boundary_gap': float(p_audio[i] - p_boundary[i]),
                    'statement_boundary_gap': float(p_statement[i] - p_boundary[i]),
                    'centroid_boundary_projection': float(p_centroid_boundary[i]),
                    'residual_boundary_projection': float(p_residual_boundary[i]),
                }
            )

        group_masks = {
            'rank_improved_or_stable': rank_stable_or_improved,
            'rank_worsened': rank_worsened,
        }
        for group, mask in group_masks.items():
            if not np.any(mask):
                continue
            group_rows.append(
                {
                    'model': model,
                    'form': form,
                    'group': group,
                    'n': int(mask.sum()),
                    'p_audio': float(np.mean(p_audio[mask])),
                    'p_boundary': float(np.mean(p_boundary[mask])),
                    'p_statement': float(np.mean(p_statement[mask])),
                    'audio_boundary_gap': float(np.mean(p_audio[mask] - p_boundary[mask])),
                    'statement_boundary_gap': float(np.mean(p_statement[mask] - p_boundary[mask])),
                    'query_move_norm': float(np.mean(move_norm[mask])),
                    'mean_rank_drop': float(np.mean(rank_drop[mask])),
                }
            )

        total_delta_ss = float(np.sum(np.linalg.norm(delta, axis=1) ** 2))
        centroid_ss = float(q_form.shape[0] * np.linalg.norm(centroid) ** 2)
        centroid_rows.append(
            {
                'model': model,
                'form': form,
                'centroid_norm': float(np.linalg.norm(centroid)),
                'variance_explained_by_centroid': centroid_ss / total_delta_ss if total_delta_ss else 0.0,
                'rankdrop_corr_query_move_norm': corr(move_norm, rank_drop),
                'rankdrop_corr_p_audio': corr(p_audio, rank_drop),
                'rankdrop_corr_p_boundary': corr(p_boundary, rank_drop),
                'rankdrop_corr_p_statement': corr(p_statement, rank_drop),
                'rankdrop_corr_centroid_boundary_projection': corr(p_centroid_boundary, rank_drop),
                'rankdrop_corr_residual_boundary_projection': corr(p_residual_boundary, rank_drop),
            }
        )

    return rows, group_rows, centroid_rows


def rankdrop_prediction_table(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    y = np.asarray([float(r['rank_drop']) for r in rows], dtype=np.float64)
    groups = np.asarray([f"{r['dataset']}::{r['audio_id']}" for r in rows])
    feature_sets = [
        ('fixed_effects_only', ()),
        ('+ query_movement_norm', ('query_move_norm',)),
        ('+ form_centroid_projection', ('centroid_boundary_projection',)),
        ('+ audio_projection', ('p_audio',)),
        ('+ boundary_projection', ('p_boundary',)),
        ('+ statement_projection', ('p_statement',)),
        ('+ audio_boundary_gap', ('audio_boundary_gap',)),
        ('+ statement_boundary_gap', ('statement_boundary_gap',)),
        (
            'full_projection_features',
            (
                'query_move_norm',
                'p_audio',
                'p_boundary',
                'p_statement',
                'audio_boundary_gap',
                'statement_boundary_gap',
                'centroid_boundary_projection',
                'residual_boundary_projection',
            ),
        ),
    ]
    if len(np.unique(groups)) < 2 or np.std(y) == 0:
        return [{'feature_set': 'insufficient_variance', 'pearson': float('nan'), 'spearman': float('nan'), 'n': len(y), 'cv': 'none'}]

    out: List[Dict[str, object]] = []
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for name, numeric_features in feature_sets:
        x_dicts: List[Dict[str, object]] = []
        for row in rows:
            item: Dict[str, object] = {
                f"dataset={row['dataset']}": 1.0,
                f"model={row['model']}": 1.0,
                f"form={row['form']}": 1.0,
            }
            for feature in numeric_features:
                item[feature] = float(row[feature])
            x_dicts.append(item)

        reg = make_pipeline(
            DictVectorizer(sparse=True),
            StandardScaler(with_mean=False),
            Ridge(alpha=1.0, solver='lsqr'),
        )
        pred = cross_val_predict(reg, x_dicts, y, groups=groups, cv=cv)
        out.append({'feature_set': name, 'pearson': corr(pred, y), 'spearman': spearman(pred, y), 'n': len(y), 'cv': 'group_kfold_audio_id'})
    return out


def weighted_group_summary(group_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_group: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in group_rows:
        by_group[str(row['group'])].append(row)
    out: List[Dict[str, object]] = []
    for group in ('rank_improved_or_stable', 'rank_worsened'):
        rows = by_group[group]
        total = sum(int(r['n']) for r in rows)
        if total == 0:
            continue
        item: Dict[str, object] = {'group': group, 'n': total}
        for key in ('p_audio', 'p_boundary', 'p_statement', 'audio_boundary_gap', 'statement_boundary_gap', 'query_move_norm', 'mean_rank_drop'):
            item[key] = sum(float(r[key]) * int(r['n']) for r in rows) / total
        out.append(item)
    return out


def dataset_group_summary(pair_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for row in pair_rows:
        group = 'rank_worsened' if int(row['rank_worsened']) else 'rank_improved_or_stable'
        by_key[(row['dataset'], group)].append(row)
    out: List[Dict[str, object]] = []
    for (dataset, group), rows in sorted(by_key.items()):
        item: Dict[str, object] = {'dataset': dataset, 'group': group, 'n': len(rows)}
        for key in ('p_audio', 'p_boundary', 'p_statement', 'audio_boundary_gap', 'statement_boundary_gap', 'query_move_norm', 'rank_drop'):
            out_key = key if key != 'rank_drop' else 'mean_rank_drop'
            item[out_key] = float(np.mean([float(r[key]) for r in rows]))
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-caption anchored CORA projection diagnostics.")
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
    if STATEMENT not in EQ_FORMS:
        raise ValueError("--query-forms must include statement because cache order validation uses it.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in EQ_JSONL.read_text(encoding='utf-8').splitlines()]
    all_rows: List[Dict[str, object]] = []
    all_group_rows: List[Dict[str, object]] = []
    all_centroid_rows: List[Dict[str, object]] = []
    for model in MODELS:
        rows, group_rows, centroid_rows = projection_rows_for_model(model, records)
        all_rows.extend(rows)
        all_group_rows.extend(group_rows)
        all_centroid_rows.extend(centroid_rows)

    prediction_rows = rankdrop_prediction_table(all_rows)
    overall_group_rows = weighted_group_summary(all_group_rows)
    dataset_group_rows = dataset_group_summary(all_rows)

    write_csv(OUT_DIR / 'source_caption_projection_pairs.csv', all_rows)
    write_csv(OUT_DIR / 'source_caption_projection_group_summary.csv', all_group_rows)
    write_csv(OUT_DIR / 'source_caption_projection_overall_group_summary.csv', overall_group_rows)
    write_csv(OUT_DIR / 'source_caption_projection_dataset_group_summary.csv', dataset_group_rows)
    write_csv(OUT_DIR / 'source_caption_projection_rankdrop_prediction.csv', prediction_rows)
    write_csv(OUT_DIR / 'source_caption_centroid_decomposition.csv', all_centroid_rows)

    pair_n = len(all_rows)
    worsen_n = sum(1 for r in all_rows if int(r['rank_drop']) > 0)
    mean_rankdrop = float(np.mean([float(r['rank_drop']) for r in all_rows]))
    lines = [
        '# Source-Caption Anchored Projection Analysis',
        '',
        'Anchor: `source_caption`; forms: EQ5.',
        '',
        'This run reports target-audio, target-negative boundary, and auxiliary statement projections against continuous RankDrop.',
        '',
        f'Pairwise rows: `{pair_n}`',
        f'Rank-worsened pairs: `{worsen_n}` ({100.0 * worsen_n / pair_n:.3f}%)',
        f'Mean RankDrop: `{mean_rankdrop:.3f}`',
        '',
        '## Projection RankDrop Prediction',
        '',
        '| feature set | Pearson r | Spearman rho | n | CV |',
        '|---|---:|---:|---:|---|',
    ]
    for row in prediction_rows:
        lines.append(f"| {row['feature_set']} | {float(row['pearson']):.3f} | {float(row['spearman']):.3f} | {int(row['n'])} | {row['cv']} |")
    lines.extend(['', '## Overall Group Means', '', '| group | p_audio | p_boundary | p_statement | audio-boundary gap | statement-boundary gap | query move | mean RankDrop | n |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for row in overall_group_rows:
        lines.append(
            '| {group} | {p_audio:.4f} | {p_boundary:.4f} | {p_statement:.4f} | {audio_boundary_gap:.4f} | {statement_boundary_gap:.4f} | {query_move_norm:.4f} | {mean_rank_drop:.3f} | {n} |'.format(**row)
        )
    lines.extend(['', '## Files', '', '- `source_caption_projection_pairs.csv`', '- `source_caption_projection_group_summary.csv`', '- `source_caption_projection_overall_group_summary.csv`', '- `source_caption_projection_dataset_group_summary.csv`', '- `source_caption_projection_rankdrop_prediction.csv`', '- `source_caption_centroid_decomposition.csv`'])
    (OUT_DIR / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {OUT_DIR}')


if __name__ == '__main__':
    main()

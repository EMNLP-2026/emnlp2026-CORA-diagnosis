# Paper Experiment Index

All public experiment entrypoints live under `scripts/`. They generate local
result tables and `REPORT.md` files under `experiments/`; they do not generate
paper figures.

| Result diagnostic | Primary script | Default output |
| --- | --- | --- |
| Source-caption anchor diagnostic | `scripts/run_source_caption_anchor_diagnostic.py` | `experiments/source_caption_anchor_diagnostic/` |
| Source-caption projection analysis | `scripts/run_source_caption_projection_analysis.py` | `experiments/source_caption_projection_analysis/` |
| TBMD-guided view intervention | `scripts/run_tbmd_guided_view_intervention.py` | `experiments/tbmd_guided_view_intervention/` |
| External retriever core diagnostic | `scripts/run_external_core_diagnostic.py` | `experiments/external_retriever_core_diagnostic/` |
| RankDrop token saliency audit | `scripts/run_rankdrop_token_saliency.py` | `experiments/rankdrop_token_saliency/` |

Historical exploratory experiments from the internal workspace are intentionally
not included in this public package.

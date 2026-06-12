# Reproducibility Notes

## Release Scope

This repository ships code for reproducing CORA paper experiment results. It
does not ship generated outputs, figures, model checkpoints, or third-party
model repositories.

The data setup command downloads only the Hugging Face `test` split:

```bash
uv run python scripts/setup_cora_test_split.py
```

Generated result artifacts are written locally under:

```text
data/
results/
experiments/
```

## Determinism

Most table regeneration is deterministic given identical embeddings. Token
saliency sampling uses `--seed` and defaults to 23. GPU kernels and third-party
model libraries can still introduce small numerical variation.

## Cache Contract

Base CLAP cache scripts write:

```text
results/eq_renew/vanilla_multiclap_eq_embedding_cache/
results/eq_renew/single_model_invariance_embedding_cache/
```

External retriever cache scripts write:

```text
results/eq_renew/external_retriever_embedding_cache/<model_label>/
```

The diagnostic scripts consume these caches and do not reload model checkpoints.

## Validation Commands

Validate downloaded data, configured model paths, and optional caches:

```bash
uv run python scripts/check_local_inputs.py
uv run python scripts/check_local_inputs.py --check-caches --check-external
```

Verify the public source layout:

```bash
uv run python scripts/verify_project.py
```

Run result diagnostics:

```bash
uv run python scripts/run_paper_experiments.py
```

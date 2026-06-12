# Model Preparation

Model code and checkpoints must be prepared manually. The setup script downloads
only the CORA test data; it does not download model weights or clone external
model repositories.

## Base CLAP Models

Default paths in `config.yaml`:

| Model | Required local inputs |
| --- | --- |
| LAION-CLAP | `models/laion-clap-htsat-unfused/` as a local Transformers model directory or local HF cache target |
| M2D-CLAP | `models_third_party/m2d/` plus `models/m2d/checkpoint-30.pth` |
| MGA-CLAP | `models_third_party/MGA-CLAP/` plus `models/mga/mga-clap.pt` |
| MS-CLAP | installed `msclap` package plus `models/msclap/msclap_2023.pt` |

Change these paths in `config.yaml` before running `scripts/embed_eq_renew_caches.py`.

## External Retrievers

External diagnostics are optional. To regenerate their caches, manually prepare:

| Model label | Required local inputs |
| --- | --- |
| `oea_qwen3b_cl` | `models_third_party/Omni-Embed-Audio/`, local OEA base model path, local LoRA/head checkpoint |
| `oea_qwen3b_ac` | same repo, AC base model path, AC LoRA/head checkpoint |
| `robustclap` | `models_third_party/linguistic_robust_clap/`, local RobustCLAP checkpoint |

Example:

```bash
uv run python scripts/generate_external_retriever_embeddings.py \
  --model robustclap \
  --model-label robustclap \
  --eq-jsonl data/cora/test/eq_by_clip.jsonl \
  --audio-dir data/cora/test/audio \
  --robustclap-repo-path models_third_party/linguistic_robust_clap \
  --robustclap-checkpoint models/robustclap/630k-audioset-best.pt
```

No script in this repo should require `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` for
model loading.

# Data Preparation

The public package downloads the CORA test split from Hugging Face and converts
it into the local layout consumed by the experiment scripts.

```bash
uv run python scripts/setup_cora_test_split.py
```

Default source and split:

```text
msnowchanj/CORA, split=test
```

The Hugging Face metadata also lists a `validation` split, but this public
reproduction flow intentionally uses only `test`.

Default local output:

```text
data/cora/test/eq_by_clip.jsonl
data/cora/test/audio/
```

The setup script writes one JSONL row per example and stores audio files under
`data/cora/test/audio/` when the Hugging Face dataset exposes an audio column.
The source dataset exposes flat query columns (`key_phrase`, `statement`,
`question`, `command`, and `indirect`). The setup script normalizes each JSONL
row for the experiment loaders to include:

- `audio_id`: stable clip identifier.
- `dataset` or `dataset_slug`: evaluated dataset pool.
- `source_caption`: caption-style anchor query.
- `generated_queries`: object with `key_phrase`, `statement`, `question`, `command`, and `indirect`.
- `metadata.relative_path` or `metadata.file_name`: local audio file reference.

Audio paths are resolved against `--audio-dir` and, if needed,
`--audio-dir/<dataset>/`.

Validate the downloaded data layout:

```bash
uv run python scripts/check_local_inputs.py --eq-jsonl data/cora/test/eq_by_clip.jsonl --audio-dir data/cora/test/audio
```

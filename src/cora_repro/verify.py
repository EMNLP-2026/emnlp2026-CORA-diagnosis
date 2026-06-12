from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .paths import PROJECT_ROOT


REQUIRED_PATHS = [
    "pyproject.toml",
    "uv.lock",
    "config.yaml",
    "README.md",
    "docs/DATA.md",
    "docs/MODELS.md",
    "docs/EXPERIMENTS.md",
    "docs/REPRODUCIBILITY.md",
    "scripts/setup_cora_test_split.py",
    "scripts/check_local_inputs.py",
    "scripts/embed_eq_renew_caches.py",
    "scripts/run_paper_experiments.py",
    "scripts/summarize_experiments.py",
    "scripts/run_source_caption_anchor_diagnostic.py",
    "scripts/run_source_caption_projection_analysis.py",
    "scripts/run_tbmd_guided_view_intervention.py",
    "scripts/generate_external_retriever_embeddings.py",
    "scripts/run_external_core_diagnostic.py",
    "scripts/run_rankdrop_token_saliency.py",
    "scripts/token_saliency_helpers.py",
    "src/clap_eval/models/laion.py",
    "src/clap_eval/models/m2d.py",
    "src/clap_eval/models/mga.py",
    "src/clap_eval/models/msclap.py",
    "src/cora_repro/summarize.py",
]


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str


def _exists(root: Path, rel: str) -> Check:
    path = root / rel
    return Check(rel, path.exists(), "exists" if path.exists() else "missing")


def collect_checks(root: Path) -> list[Check]:
    return [_exists(root, rel) for rel in REQUIRED_PATHS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the public CORA code package layout.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    checks = collect_checks(args.root)
    ok = all(check.ok for check in checks)
    if args.json:
        print(json.dumps({"ok": ok, "checks": [check.__dict__ for check in checks]}, indent=2))
    else:
        print(f"CORA public code package root: {args.root}")
        for check in checks:
            status = "OK" if check.ok else "MISSING"
            print(f"[{status}] {check.name} -- {check.evidence}")
        print(f"\nOverall: {'OK' if ok else 'MISSING REQUIREMENTS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

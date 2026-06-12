#!/usr/bin/env python3
"""Run paper-facing CORA diagnostics and report result artifact locations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-facing CORA diagnostics from local embedding caches.")
    parser.add_argument("--eq-jsonl", type=Path, default=PROJECT_ROOT / "data/cora/test/eq_by_clip.jsonl")
    parser.add_argument("--base-cache-dir", type=Path, default=PROJECT_ROOT / "results/eq_renew/vanilla_multiclap_eq_embedding_cache")
    parser.add_argument("--source-cache-dir", type=Path, default=PROJECT_ROOT / "results/eq_renew/single_model_invariance_embedding_cache")
    parser.add_argument("--include-external", action="store_true", help="Also run optional OEA/RobustCLAP diagnostics from external caches.")
    parser.add_argument("--include-token-saliency", action="store_true", help="Also run the slower RankDrop token-saliency audit.")
    parser.add_argument("--skip-input-check", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def print_report(report: Path) -> None:
    rel = report.relative_to(PROJECT_ROOT)
    print(f"\n===== {rel} =====")
    if not report.exists():
        print("[missing report]")
        return
    print(report.read_text(encoding="utf-8").strip())


def main() -> int:
    args = parse_args()
    required = [args.eq_jsonl, args.base_cache_dir, args.source_cache_dir]
    missing = [path for path in required if not path.exists()]
    if missing and not args.skip_input_check:
        print("Missing required inputs:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        print("\nRun scripts/setup_cora_test_split.py, prepare model checkpoints, and build embedding caches first.", file=sys.stderr)
        return 2

    common = [
        "--eq-jsonl", str(args.eq_jsonl),
        "--base-cache-dir", str(args.base_cache_dir),
        "--source-cache-dir", str(args.source_cache_dir),
    ]
    jobs: list[tuple[str, list[str], Path]] = [
        (
            "source-caption anchor diagnostic",
            [sys.executable, "scripts/run_source_caption_anchor_diagnostic.py", *common],
            PROJECT_ROOT / "experiments/source_caption_anchor_diagnostic/REPORT.md",
        ),
        (
            "source-caption projection analysis",
            [sys.executable, "scripts/run_source_caption_projection_analysis.py", *common],
            PROJECT_ROOT / "experiments/source_caption_projection_analysis/REPORT.md",
        ),
        (
            "TBMD-guided view intervention",
            [sys.executable, "scripts/run_tbmd_guided_view_intervention.py", *common],
            PROJECT_ROOT / "experiments/tbmd_guided_view_intervention/REPORT.md",
        ),
    ]
    if args.include_external:
        jobs.append(
            (
                "external retriever core diagnostic",
                [sys.executable, "scripts/run_external_core_diagnostic.py"],
                PROJECT_ROOT / "experiments/external_retriever_core_diagnostic/REPORT.md",
            )
        )
    if args.include_token_saliency:
        jobs.append(
            (
                "RankDrop token saliency audit",
                [sys.executable, "scripts/run_rankdrop_token_saliency.py"],
                PROJECT_ROOT / "experiments/rankdrop_token_saliency/REPORT.md",
            )
        )

    reports: list[Path] = []
    for _, cmd, report in jobs:
        run(cmd)
        reports.append(report)

    for report in reports:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

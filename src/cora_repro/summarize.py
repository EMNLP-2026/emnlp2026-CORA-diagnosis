from __future__ import annotations

import argparse
from pathlib import Path

from .paths import PROJECT_ROOT


def first_heading(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except UnicodeDecodeError:
        return path.name
    return path.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a compact index of generated CORA experiment artifacts.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)

    exp_root = args.root / "experiments"
    if not exp_root.exists():
        print(f"No generated experiment directory found: {exp_root}")
        return 0

    exp_dirs = sorted(path for path in exp_root.iterdir() if path.is_dir())
    if not exp_dirs:
        print(f"No generated experiment outputs found under: {exp_root}")
        return 0

    for exp_dir in exp_dirs:
        doc = exp_dir / "README.md"
        if not doc.exists():
            doc = exp_dir / "REPORT.md"
        files = [path for path in exp_dir.iterdir() if path.is_file()]
        csv_count = sum(1 for path in files if path.suffix == ".csv")
        json_count = sum(1 for path in files if path.suffix == ".json")
        md_count = sum(1 for path in files if path.suffix == ".md")
        print(f"{exp_dir.name}: {first_heading(doc)}")
        print(f"  files={len(files)} csv={csv_count} json={json_count} md={md_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

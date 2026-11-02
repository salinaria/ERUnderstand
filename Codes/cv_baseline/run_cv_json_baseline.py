#!/usr/bin/env python3
"""
Run the OpenCV ER baseline on a subset of diagrams and write benchmark JSON.

Synthetic easy / med / hard use 50 evenly spaced diagrams each. Real-world
tiers use every other diagram (about half). Output matches the other models:

  Experiments/Graphviz/Graphviz-{easy,med,hard}/cv_baseline/<stem>.json
  Experiments/Real-world/<tier>/cv_baseline/<stem>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from erd_analyzer_shape_based_v2 import ERDAnalyzer
from erd_json import diagram_to_json
from local_text_extractor import LocalTextExtractor

REPO_ROOT = _SCRIPT_DIR.parent.parent

# (tier name, image directory, prediction directory, subset size or None for half)
TIER_SPECS = [
    (
        "easy",
        REPO_ROOT / "ERUnderstand/Generated/Graphviz/Graphviz-easy",
        REPO_ROOT / "Experiments/Graphviz/Graphviz-easy/cv_baseline",
        50,
    ),
    (
        "med",
        REPO_ROOT / "ERUnderstand/Generated/Graphviz/Graphviz-med",
        REPO_ROOT / "Experiments/Graphviz/Graphviz-med/cv_baseline",
        50,
    ),
    (
        "hard",
        REPO_ROOT / "ERUnderstand/Generated/Graphviz/Graphviz-hard",
        REPO_ROOT / "Experiments/Graphviz/Graphviz-hard/cv_baseline",
        50,
    ),
    (
        "Web",
        REPO_ROOT / "ERUnderstand/Real-world/Diagrams/Web",
        REPO_ROOT / "Experiments/Real-world/Web/cv_baseline",
        None,
    ),
    (
        "Web-Graphviz",
        REPO_ROOT / "ERUnderstand/Real-world/Diagrams/Web-Graphviz",
        REPO_ROOT / "Experiments/Real-world/Web-Graphviz/cv_baseline",
        None,
    ),
    (
        "Web-Silber",
        REPO_ROOT / "ERUnderstand/Real-world/Diagrams/Web-Silber",
        REPO_ROOT / "Experiments/Real-world/Web-Silber/cv_baseline",
        None,
    ),
    (
        "Instructional",
        REPO_ROOT / "ERUnderstand/Real-world/Diagrams/Instructional",
        REPO_ROOT / "Experiments/Real-world/Instructional/cv_baseline",
        None,
    ),
    (
        "Instructional-Graphviz",
        REPO_ROOT / "ERUnderstand/Real-world/Diagrams/Instructional-Graphviz",
        REPO_ROOT / "Experiments/Real-world/Instructional-Graphviz/cv_baseline",
        None,
    ),
    (
        "Instructional-Silber",
        REPO_ROOT / "ERUnderstand/Real-world/Diagrams/Instructional-Silber",
        REPO_ROOT / "Experiments/Real-world/Instructional-Silber/cv_baseline",
        None,
    ),
    (
        "Schema",
        REPO_ROOT / "ERUnderstand/Real-world/Diagrams/Schema-Graphviz",
        REPO_ROOT / "Experiments/Real-world/Schema/cv_baseline",
        None,
    ),
    (
        "med-random",
        REPO_ROOT / "ERUnderstand/Generated/Graphviz/Graphviz-med-random",
        REPO_ROOT / "Experiments/Graphviz/Graphviz-med-random/cv_baseline",
        199,
    ),
    (
        "med-biased",
        REPO_ROOT / "ERUnderstand/Generated/Graphviz/Graphviz-med-biased",
        REPO_ROOT / "Experiments/Graphviz/Graphviz-med-biased/cv_baseline",
        199,
    ),
]


def list_images(folder: Path) -> list[Path]:
    """Sorted diagram images in one tier folder."""
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def select_subset(images: list[Path], count: int | None) -> list[Path]:
    """Pick ``count`` evenly spaced images, or every other image when count is None."""
    if not images:
        return []
    if count is None:
        return images[::2]
    if count >= len(images):
        return images
    if count <= 1:
        return images[:count]
    chosen = []
    seen = set()
    last = len(images) - 1
    for index in range(count):
        position = round(index * last / (count - 1))
        if position not in seen:
            seen.add(position)
            chosen.append(images[position])
    return chosen


def run_tier(name: str, image_dir: Path, out_dir: Path, count: int | None, extractor, analyzer) -> tuple[int, int]:
    """Write one JSON per selected diagram. Returns (written, failed)."""
    selected = select_subset(list_images(image_dir), count)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name}: {len(selected)} diagrams from {image_dir.name} ===", flush=True)
    written = failed = 0
    for index, image_path in enumerate(selected, start=1):
        out_path = out_dir / f"{image_path.stem}.json"
        if out_path.exists():
            continue
        print(f"[{name} {index}/{len(selected)}] {image_path.name}", flush=True)
        try:
            result = diagram_to_json(image_path, extractor=extractor, analyzer=analyzer)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written += 1
        except Exception:
            failed += 1
            print(f"[FAIL] {image_path.name}", flush=True)
            traceback.print_exc()
    return written, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="CV baseline ERD→JSON subset experiments")
    parser.add_argument(
        "--tiers",
        nargs="*",
        default=None,
        help="Tier names to run (default: all). Example: easy med hard Web",
    )
    args = parser.parse_args()

    wanted = set(args.tiers) if args.tiers else None
    extractor = LocalTextExtractor()
    analyzer = ERDAnalyzer(use_clip=False)
    total_written = total_failed = 0
    for name, image_dir, out_dir, count in TIER_SPECS:
        if wanted is not None and name not in wanted:
            continue
        written, failed = run_tier(name, image_dir, out_dir, count, extractor, analyzer)
        total_written += written
        total_failed += failed
    print(f"\nDone. wrote {total_written}, failed {total_failed}", flush=True)


if __name__ == "__main__":
    main()

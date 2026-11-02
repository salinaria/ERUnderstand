#!/usr/bin/env python3
"""
Batch-run erd_analyzer_shape_based.py on the Error New dataset.

Ground truth for erroneous diagrams: folder name TYPE_{code} under
Error New/Error Dataset/Images/{subdir}/.
Optional correct (error-free) PNGs from Graphviz-easy for false-positive rate.

Outputs CSV + JSON under Error New/Eval Results/cv_shape_analyzer/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow imports from this directory when run from repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# cv_baseline lives at ``Error New/src/cv_baseline`` → git repo root is three parents.
REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
DEFAULT_IMAGES = REPO_ROOT / "Error New/Error Dataset/Images" / "easy"
DEFAULT_CORRECT = REPO_ROOT / "Error New/Dataset/Generated/Graphviz/Graphviz-easy"
ERROR_KEYS = [f"S{i}" for i in range(1, 7)] + [f"C{i}" for i in range(1, 6)]
RESULTS_BASE = REPO_ROOT / "Error New/Eval Results"

# Lazy import after sys.path setup (see ``_ocr_pairing``).
_OCR_EXTRACTOR = None


def _load_analyzer(version: str):
    """Import ERDAnalyzer + ERROR_LABELS for v1 or v2."""
    if version == "v1":
        from erd_analyzer_shape_based import ERDAnalyzer, ERROR_LABELS
    elif version == "v2":
        from erd_analyzer_shape_based_v2 import ERDAnalyzer, ERROR_LABELS
    else:
        raise ValueError(f"Unknown analyzer version: {version!r} (use v1 or v2)")
    return ERDAnalyzer, ERROR_LABELS


def _limit_pngs(pngs: List[Path], max_per_class: Optional[int]) -> List[Path]:
    if max_per_class is None:
        return pngs
    return pngs[:max_per_class]


def load_samples(
    images_base: Path,
    include_correct: bool,
    max_per_class: Optional[int],
    correct_dir: Optional[Path],
    max_total: Optional[int] = None,
) -> List[Tuple[Path, str]]:
    """Return (image_path, ground_truth_key) pairs."""
    by_key: Dict[str, List[Path]] = {}
    for key in ERROR_KEYS:
        folder = images_base / f"TYPE_{key}"
        if not folder.is_dir():
            print(f"  [loader] WARNING – missing folder: {folder}")
            continue
        by_key[key] = _limit_pngs(sorted(folder.glob("*.png")), max_per_class)

    if include_correct:
        correct_folder = correct_dir if correct_dir is not None else DEFAULT_CORRECT
        if correct_folder.is_dir():
            by_key["CORRECT"] = _limit_pngs(sorted(correct_folder.glob("*.png")), max_per_class)
        else:
            print(f"  [loader] WARNING – correct folder not found: {correct_folder}")

    if max_total is None:
        samples: List[Tuple[Path, str]] = []
        for key in ERROR_KEYS + (["CORRECT"] if "CORRECT" in by_key else []):
            for p in by_key.get(key, []):
                samples.append((p, key))
        return samples

    # Round-robin across classes so the cap is balanced.
    keys = [k for k in ERROR_KEYS if k in by_key]
    if "CORRECT" in by_key:
        keys.append("CORRECT")
    samples = []
    indices = {k: 0 for k in keys}
    while len(samples) < max_total:
        added = False
        for key in keys:
            paths = by_key[key]
            i = indices[key]
            if i < len(paths):
                samples.append((paths[i], key))
                indices[key] = i + 1
                added = True
                if len(samples) >= max_total:
                    break
        if not added:
            break
    return samples


def predicted_codes(result, error_labels: Dict[str, str]) -> Set[str]:
    """Unique S/C error codes reported by the analyzer."""
    return {e.code for e in result.errors if e.code in error_labels}


def _ocr_pairing():
    from local_text_extractor import LocalTextExtractor
    from text_node_pairing import elements_by_error_code, pair_ocr_labels_to_nodes

    global _OCR_EXTRACTOR
    if _OCR_EXTRACTOR is None:
        _OCR_EXTRACTOR = LocalTextExtractor()
    return _OCR_EXTRACTOR, pair_ocr_labels_to_nodes, elements_by_error_code


def primary_prediction(codes: Set[str], gt_key: str) -> Optional[str]:
    """
    Pick one predicted type for scoring.
    Prefer the ground-truth family (S vs C) when both appear.
    """
    if not codes:
        return None
    if gt_key in codes:
        return gt_key
    if gt_key.startswith("S"):
        s_codes = sorted(c for c in codes if c.startswith("S"))
        if s_codes:
            return s_codes[0]
    if gt_key.startswith("C"):
        c_codes = sorted(c for c in codes if c.startswith("C"))
        if c_codes:
            return c_codes[0]
    return sorted(codes)[0]


def _element_overlap(gt_elements: str, pred_elements: str) -> bool:
    """Token/substring overlap (aligned with ``erd_eval_results_analysis``)."""
    if not str(gt_elements or "").strip():
        return False
    gt = str(gt_elements).strip().lower()
    pred = str(pred_elements or "").strip().lower()
    if not pred:
        return False
    gt_toks = {t for t in gt.replace(",", " ").replace(";", " ").split() if t}
    pred_toks = {t for t in pred.replace(",", " ").replace(";", " ").split() if t}
    if gt_toks & pred_toks:
        return True
    return any(g in pred for g in gt_toks)


def _load_gt_elements(image_rel: str) -> str:
    """Read ``Elements`` from sibling JSON under Error Dataset/JSON."""
    parts = Path(image_rel).parts
    if "Images" not in parts:
        return ""
    json_parts = tuple("JSON" if seg == "Images" else seg for seg in parts)
    jp = REPO_ROOT.joinpath(*json_parts).with_suffix(".json")
    if not jp.is_file():
        return ""
    try:
        data = json.loads(jp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("Elements") or "").strip()
    if isinstance(data, dict):
        return str(data.get("Elements") or "").strip()
    return ""


def summarize(rows: List[dict], error_labels: Dict[str, str]) -> dict:
    """Aggregate accuracy metrics aligned with erd_eval studies."""
    n = len(rows)
    if n == 0:
        return {}

    binary_ok = sum(1 for r in rows if r["binary_correct"])
    type_ok = sum(1 for r in rows if r["type_correct"])
    type_in_set = sum(1 for r in rows if r["gt_in_predicted_set"])

    by_gt: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_gt[r["gt"]].append(r)

    per_class = {}
    for gt, items in sorted(by_gt.items()):
        per_class[gt] = {
            "n": len(items),
            "binary_acc": sum(1 for x in items if x["binary_correct"]) / len(items),
            "type_acc": sum(1 for x in items if x["type_correct"]) / len(items),
            "gt_in_set_acc": sum(1 for x in items if x["gt_in_predicted_set"]) / len(items),
        }

    err_rows = [r for r in rows if r["gt"] != "CORRECT"]
    correct_rows = [r for r in rows if r["gt"] == "CORRECT"]

    fp = sum(1 for r in correct_rows if r["pred_has_error"])
    fn = sum(1 for r in err_rows if not r["pred_has_error"])

    elem_ok = 0
    elem_n = 0
    for r in err_rows:
        gt_el = _load_gt_elements(r["image"])
        if not gt_el:
            continue
        scoped = (r.get("pred_elements_by_code") or {}).get(r["gt"], "")
        elem_n += 1
        if _element_overlap(gt_el, scoped):
            elem_ok += 1

    element_accuracy = (elem_ok / elem_n) if elem_n else None

    return {
        "n_total": n,
        "n_erroneous": len(err_rows),
        "n_correct": len(correct_rows),
        "binary_accuracy": binary_ok / n,
        "type_accuracy_exact_primary": type_ok / n,
        "type_accuracy_gt_in_set": type_in_set / n,
        "false_positives_on_correct": fp,
        "false_negatives_on_erroneous": fn,
        "element_accuracy_scoped_gt": element_accuracy,
        "element_n_evaluable": elem_n,
        "predicted_code_counts": dict(Counter(r["pred_primary"] for r in rows if r["pred_primary"])),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ERD shape-based CV analyzer")
    parser.add_argument(
        "--images-base",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Directory containing TYPE_S1 … TYPE_C5 folders",
    )
    parser.add_argument(
        "--correct-dir",
        type=Path,
        default=DEFAULT_CORRECT,
        help="Error-free PNGs (CORRECT class)",
    )
    parser.add_argument("--subdir", default="", help="Unused if --images-base is explicit")
    parser.add_argument("--max-per-class", type=int, default=None, help="Cap PNGs per folder")
    parser.add_argument("--max-total", type=int, default=None, help="Cap total images (balanced round-robin)")
    parser.add_argument("--no-correct", action="store_true", help="Skip CORRECT / Graphviz-easy set")
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Run only error-free PNGs from --correct-dir (Graphviz-easy)",
    )
    parser.add_argument("--no-clip", action="store_true", help="Geometry-only shape classification")
    parser.add_argument(
        "--no-shape-error-inference",
        action="store_true",
        help="Only direct connection errors (no S-code heuristics)",
    )
    parser.add_argument("--save-debug", action="store_true", help="Write debug PNGs (slow, large)")
    parser.add_argument(
        "--ocr-elements",
        action="store_true",
        default=True,
        help="Use local EasyOCR to label shapes for Elements fields (default: on)",
    )
    parser.add_argument(
        "--no-ocr-elements",
        action="store_false",
        dest="ocr_elements",
        help="Skip OCR; leave Elements empty in exported study CSVs",
    )
    parser.add_argument(
        "--ocr-language",
        default="english",
        help="Language key for LocalTextExtractor (default: english)",
    )
    parser.add_argument(
        "--analyzer",
        choices=["v1", "v2"],
        default="v1",
        help="Analyzer module: erd_analyzer_shape_based (v1) or _v2 (v2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Results directory (default: cv_shape_analyzer_{v1|v2})",
    )
    args = parser.parse_args()

    ERDAnalyzer, ERROR_LABELS = _load_analyzer(args.analyzer)
    if args.output_dir is None:
        args.output_dir = RESULTS_BASE / f"cv_shape_analyzer_{args.analyzer}"

    images_base = args.images_base
    if args.subdir and args.images_base == DEFAULT_IMAGES:
        images_base = REPO_ROOT / "Error New/Error Dataset/Images" / args.subdir

    if args.correct_only and args.no_correct:
        raise SystemExit("Use only one of --correct-only or --no-correct.")

    if args.correct_only:
        correct_folder = args.correct_dir
        if not correct_folder.is_dir():
            raise SystemExit(f"Correct folder not found: {correct_folder}")
        samples = [
            (p, "CORRECT")
            for p in _limit_pngs(sorted(correct_folder.glob("*.png")), args.max_per_class)
        ]
        print(f"Loaded {len(samples)} CORRECT images from {correct_folder}")
    else:
        samples = load_samples(
            images_base,
            include_correct=not args.no_correct,
            max_per_class=args.max_per_class,
            correct_dir=args.correct_dir,
            max_total=args.max_total,
        )
        print(f"Loaded {len(samples)} images from {images_base}")
        if not args.no_correct:
            print(f"  (+ CORRECT pool: {args.correct_dir})")

    analyzer = ERDAnalyzer(
        use_clip=not args.no_clip,
        infer_shape_errors=not args.no_shape_error_inference,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_dir / "debug" if args.save_debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    ocr_extractor = None
    pair_fn = None
    elements_fn = None
    if args.ocr_elements:
        try:
            ocr_extractor, pair_fn, elements_fn = _ocr_pairing()
            print(f"  [OCR] EasyOCR enabled for Elements pairing ({args.ocr_language})")
        except Exception as exc:
            print(f"  [OCR] Disabled — could not load EasyOCR: {exc}")

    rows: List[dict] = []
    t0 = time.time()

    for i, (img_path, gt) in enumerate(samples, start=1):
        debug_path = None
        if debug_dir:
            debug_path = debug_dir / f"{gt}_{img_path.stem}.png"

        result = analyzer.analyze(img_path, save_debug_to=debug_path)

        pred_elements_by_code: Dict[str, str] = {}
        if pair_fn is not None and elements_fn is not None and result.ok:
            try:
                ocr_cache = args.output_dir / "ocr_cache"
                pair_fn(
                    img_path,
                    result.nodes,
                    language=args.ocr_language,
                    extractor=ocr_extractor,
                    cache_dir=ocr_cache,
                )
                pred_elements_by_code = elements_fn(result.errors, result.nodes)
            except Exception as exc:
                result.issues.append(f"ocr_pairing: {exc}")

        codes = predicted_codes(result, ERROR_LABELS)
        pred_has_error = len(codes) > 0
        gt_has_error = gt != "CORRECT"
        pred_primary = primary_prediction(codes, gt) if pred_has_error else None
        pred_elements_scoped = (
            pred_elements_by_code.get(gt, "") if gt_has_error else ""
        )

        row = {
            "image": str(img_path.resolve().relative_to(REPO_ROOT)),
            "stem": img_path.stem,
            "gt": gt,
            "pred_has_error": pred_has_error,
            "pred_codes": sorted(codes),
            "pred_primary": pred_primary,
            "pred_elements": pred_elements_scoped,
            "pred_elements_by_code": pred_elements_by_code,
            "n_nodes": len(result.nodes),
            "n_edges": len(result.edges),
            "binary_correct": pred_has_error == gt_has_error,
            "type_correct": (pred_primary == gt) if gt_has_error else (not pred_has_error),
            "gt_in_predicted_set": (gt in codes) if gt_has_error else (len(codes) == 0),
            "ok": result.ok,
            "issues": result.issues,
        }
        rows.append(row)

        if i % 100 == 0 or i == len(samples):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"  [{i}/{len(samples)}] {rate:.1f} img/s — last: {img_path.name} GT={gt} pred={sorted(codes)}")

    summary = summarize(rows, ERROR_LABELS)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary["config"] = {
        "analyzer": args.analyzer,
        "images_base": str(images_base),
        "use_clip": not args.no_clip,
        "infer_shape_errors": not args.no_shape_error_inference,
        "ocr_elements": args.ocr_elements and pair_fn is not None,
        "ocr_language": args.ocr_language,
        "max_per_class": args.max_per_class,
        "max_total": args.max_total,
        "include_correct": not args.no_correct,
        "correct_only": args.correct_only,
        "correct_dir": str(args.correct_dir) if args.correct_only else None,
    }

    csv_path = args.output_dir / "results.csv"
    json_path = args.output_dir / "results.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                **r,
                "pred_codes": "|".join(r["pred_codes"]),
                "pred_elements_by_code": json.dumps(r.get("pred_elements_by_code") or {}),
            })

    json_rows: List[Dict[str, Any]] = []
    for r in rows:
        json_rows.append({**r, "pred_codes": list(r["pred_codes"])})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": json_rows}, f, indent=2)

    print("\n=== CV shape analyzer summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

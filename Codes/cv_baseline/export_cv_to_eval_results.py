#!/usr/bin/env python3
"""
Export CV v2 batch results into VLM-compatible study CSVs under
``Error New/Eval Results/cv_v2/`` for notebook comparison (5 studies).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_CODES = _SCRIPT_DIR.parent / "Codes"
if str(_CODES) not in sys.path:
    sys.path.insert(0, str(_CODES))

from erd_eval_results_analysis import STUDY_DIRS, _read_csv_rows  # noqa: E402

REPO_ROOT = _SCRIPT_DIR.parent.parent
EVAL_RESULTS = REPO_ROOT / "Error New/Eval Results"
SHAPE_KEYS = [f"S{i}" for i in range(1, 7)]
CONNECTION_KEYS = [f"C{i}" for i in range(1, 6)]
ALL_ERROR_KEYS = SHAPE_KEYS + CONNECTION_KEYS

TIER_DEFAULTS = {
    "easy": {
        "images_subdir": "easy",
        "correct_dir": REPO_ROOT / "Error New/Dataset/Generated/Graphviz/Graphviz-easy",
        "cv_json": EVAL_RESULTS / "cv_shape_analyzer_v2/results.json",
        "out_root": EVAL_RESULTS / "cv_v2",
        "results_subdir": "",
    },
    "med": {
        "images_subdir": "med",
        "correct_dir": REPO_ROOT / "Error New/Dataset/Generated/Graphviz/Graphviz-med",
        "cv_json": EVAL_RESULTS / "cv_shape_analyzer_v2_med/results.json",
        "out_root": EVAL_RESULTS / "med" / "cv_v2",
        "results_subdir": "med",
    },
}

# Set by ``configure_tier`` before export.
IMAGES_BASE_DIR = REPO_ROOT / "Error New/Error Dataset/Images/easy"
CORRECT_DIR = REPO_ROOT / "Error New/Dataset/Generated/Graphviz/Graphviz-easy"
CV_ERR_JSON = EVAL_RESULTS / "cv_shape_analyzer_v2/results.json"
CV_CORRECT_JSON = EVAL_RESULTS / "cv_shape_analyzer_v2_correct/results.json"
OUT_ROOT = EVAL_RESULTS / "cv_v2"
RESULTS_SUBDIR = ""
REF_MODEL = "gemma3"


def configure_tier(
    tier: str,
    *,
    cv_json: Optional[Path] = None,
    out_root: Optional[Path] = None,
    ref_model: Optional[str] = None,
) -> None:
    """Point export paths at easy or med (or overrides)."""
    global IMAGES_BASE_DIR, CORRECT_DIR, CV_ERR_JSON, OUT_ROOT, RESULTS_SUBDIR, REF_MODEL
    if tier not in TIER_DEFAULTS:
        raise ValueError(f"tier must be one of {list(TIER_DEFAULTS)}, got {tier!r}")
    cfg = TIER_DEFAULTS[tier]
    IMAGES_BASE_DIR = REPO_ROOT / "Error New/Error Dataset/Images" / cfg["images_subdir"]
    CORRECT_DIR = cfg["correct_dir"]
    CV_ERR_JSON = cv_json or cfg["cv_json"]
    OUT_ROOT = out_root or cfg["out_root"]
    RESULTS_SUBDIR = cfg["results_subdir"]
    REF_MODEL = ref_model or "gemma3"


def matched_correct_error_pairs(
    error_key: str,
    max_per_class: int = 50,
    max_pairs: int = 50,
) -> List[Tuple[Path, Path]]:
    """(correct_path, errored_path) with matching basename."""
    err_folder = IMAGES_BASE_DIR / f"TYPE_{error_key}"
    if not err_folder.is_dir() or not CORRECT_DIR.is_dir():
        return []
    err_pngs = sorted(err_folder.glob("*.png"))[:max_per_class]
    correct_by_stem = {p.stem: p for p in sorted(CORRECT_DIR.glob("*.png"))}
    out: List[Tuple[Path, Path]] = []
    for errored in err_pngs:
        c = correct_by_stem.get(errored.stem)
        if c is not None:
            out.append((c, errored))
    return out[:max_pairs]


MODEL_KEY = "cv_v2"


def _norm_image(path_str: str) -> str:
    return str(path_str).replace("\\", "/").strip()


def load_cv_lookup() -> Dict[str, dict]:
    """Map normalized image path -> CV row dict (main v2 batch; optional legacy correct-only file)."""
    lookup: Dict[str, dict] = {}
    sources: List[Path] = []
    if CV_ERR_JSON.is_file():
        sources.append(CV_ERR_JSON)
    elif CV_CORRECT_JSON.is_file():
        sources.append(CV_CORRECT_JSON)
    # Legacy separate correct-only run — only merge if main batch lacks CORRECT rows.
    if CV_CORRECT_JSON.is_file() and CV_ERR_JSON.is_file():
        main = json.loads(CV_ERR_JSON.read_text(encoding="utf-8"))
        has_correct = any((r.get("gt") or "").upper() == "CORRECT" for r in main.get("rows", []))
        if not has_correct:
            sources.append(CV_CORRECT_JSON)
    for jp in sources:
        data = json.loads(jp.read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            lookup[_norm_image(row["image"])] = row
    return lookup


def _parse_codes(row: dict) -> Set[str]:
    codes = row.get("pred_codes")
    if isinstance(codes, list):
        return {c for c in codes if c}
    if isinstance(codes, str):
        return {c for c in codes.split("|") if c}
    return set()


def _parse_elements_by_code(row: dict) -> Dict[str, str]:
    """Per-error-code OCR element strings from CV batch (rule-based type, OCR elements)."""
    raw = row.get("pred_elements_by_code")
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            pass
    return {}


def _scoped_elements(cv: dict, gt: str, codes: Set[str], *, use_pred_primary: bool = False) -> str:
    """
    Elements for export: OCR labels on shapes involved in the scoped GT error.

    When ``use_pred_primary`` and GT is not in ``codes``, use the primary predicted
    code's elements (study 4 style).
    """
    if gt == "CORRECT":
        return ""
    by_code = _parse_elements_by_code(cv)
    if gt in by_code and by_code[gt]:
        return by_code[gt]
    if use_pred_primary and codes:
        primary = _primary(codes, gt)
        return by_code.get(primary, "")
    return str(cv.get("pred_elements") or "").strip()


def _primary(codes: Set[str], gt: str) -> str:
    if not codes:
        return ""
    if gt in codes:
        return gt
    if gt.startswith("S"):
        s = sorted(c for c in codes if c.startswith("S"))
        if s:
            return s[0]
    if gt.startswith("C"):
        c = sorted(c for c in codes if c.startswith("C"))
        if c:
            return c[0]
    return sorted(codes)[0]


def _rel(path: Path) -> str:
    try:
        return _norm_image(str(path.relative_to(REPO_ROOT)))
    except ValueError:
        return _norm_image(str(path))


def _cv_row_base() -> dict:
    return {
        "model_key": MODEL_KEY,
        "model_name": "erd_analyzer_shape_based_v2",
        "backend": "cv",
    }


def _cv_gt(cv: dict) -> str:
    return (cv.get("gt") or "").strip().upper()


def _row_study1(img: str, gt: str, codes: Set[str], cv: dict) -> dict:
    has = gt in codes
    return {
        **_cv_row_base(),
        "image": img,
        "ground_truth": gt,
        "has_error": str(has),
        "TYPE": gt if has else _primary(codes, gt),
        "Elements": _scoped_elements(cv, gt, codes) if has else "",
        "raw_response": json.dumps({
            "pred_codes": sorted(codes),
            "pred_elements": _scoped_elements(cv, gt, codes),
        }),
    }


def export_study1_from_cv(lookup: Dict[str, dict]) -> List[dict]:
    """Study 1: erroneous images only (scoped binary per folder)."""
    rows: List[dict] = []
    for img in sorted(lookup):
        cv = lookup[img]
        gt = _cv_gt(cv)
        if gt == "CORRECT" or gt not in ALL_ERROR_KEYS:
            continue
        rows.append(_row_study1(img, gt, _parse_codes(cv), cv))
    return rows


def export_study1(lookup: Dict[str, dict], template_rows: List[dict]) -> List[dict]:
    if not template_rows:
        return export_study1_from_cv(lookup)
    rows: List[dict] = []
    for t in template_rows:
        img = _norm_image(t.get("image", ""))
        gt = (t.get("ground_truth") or "").strip().upper()
        cv = lookup.get(img)
        if not cv or gt == "CORRECT":
            continue
        rows.append(_row_study1(img, gt, _parse_codes(cv), cv))
    return rows


def _row_study2_or_3(
    img: str, gt: str, codes: Set[str], connection: bool, cv: dict
) -> dict:
    allowed = CONNECTION_KEYS if connection else SHAPE_KEYS
    flag = "has_connection_error" if connection else "has_shape_error"
    etype = "error_type"
    eel = "error_elements" if connection else "error_element"
    scoped = {c for c in codes if c in allowed}
    if gt == "CORRECT":
        has = bool(scoped)
        pred = ""
        elements = ""
    else:
        has = bool(scoped)
        pred = gt if gt in scoped else (_primary(scoped, gt) if scoped else "")
        elements = _scoped_elements(cv, gt, scoped) if gt in scoped else ""
    return {
        **_cv_row_base(),
        "image": img,
        "ground_truth": gt,
        flag: str(has),
        etype: pred,
        eel: elements,
        "raw_response": json.dumps({
            "pred_codes": sorted(codes),
            "pred_elements": elements,
        }),
    }


def export_study2_or_3_from_cv(lookup: Dict[str, dict], connection: bool) -> List[dict]:
    """All CV images for shape (S*) or connection (C*) study + CORRECT."""
    allowed_gt = set(CONNECTION_KEYS if connection else SHAPE_KEYS) | {"CORRECT"}
    rows: List[dict] = []
    for img in sorted(lookup):
        cv = lookup[img]
        gt = _cv_gt(cv)
        if gt not in allowed_gt:
            continue
        rows.append(_row_study2_or_3(img, gt, _parse_codes(cv), connection, cv))
    return rows


def export_study2_or_3(lookup: Dict[str, dict], template_rows: List[dict], connection: bool) -> List[dict]:
    if not template_rows:
        return export_study2_or_3_from_cv(lookup, connection)
    rows: List[dict] = []
    for t in template_rows:
        img = _norm_image(t.get("image", ""))
        gt = (t.get("ground_truth") or "").strip().upper()
        cv = lookup.get(img)
        if not cv:
            continue
        rows.append(_row_study2_or_3(img, gt, _parse_codes(cv), connection, cv))
    return rows


def export_study4_from_cv(lookup: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for img in sorted(lookup):
        cv = lookup[img]
        gt = _cv_gt(cv)
        codes = _parse_codes(cv)
        if gt == "CORRECT":
            has = len(codes) > 0
            pred = ""
        else:
            has = len(codes) > 0
            pred = gt if gt in codes else (_primary(codes, gt) if codes else "")
        rows.append({
            **_cv_row_base(),
            "image": img,
            "ground_truth": gt,
            "has_error": str(has),
            "TYPE": pred,
            "Elements": _scoped_elements(cv, gt, codes, use_pred_primary=True),
            "raw_response": json.dumps({
                "pred_codes": sorted(codes),
                "pred_elements": _scoped_elements(cv, gt, codes, use_pred_primary=True),
            }),
        })
    return rows


def export_study4(lookup: Dict[str, dict], template_rows: List[dict]) -> List[dict]:
    if not template_rows:
        return export_study4_from_cv(lookup)
    rows: List[dict] = []
    for t in template_rows:
        img = _norm_image(t.get("image", ""))
        gt = (t.get("ground_truth") or "").strip().upper()
        cv = lookup.get(img)
        if not cv:
            continue
        codes = _parse_codes(cv)
        if gt == "CORRECT":
            has = len(codes) > 0
            pred = ""
        else:
            has = len(codes) > 0
            pred = gt if gt in codes else (_primary(codes, gt) if codes else "")
        rows.append({
            **_cv_row_base(),
            "image": img,
            "ground_truth": gt,
            "has_error": str(has),
            "TYPE": pred,
            "Elements": _scoped_elements(cv, gt, codes, use_pred_primary=True),
            "raw_response": json.dumps({
                "pred_codes": sorted(codes),
                "pred_elements": _scoped_elements(cv, gt, codes, use_pred_primary=True),
            }),
        })
    return rows


def _errored_side_codes(
    path_a: str, path_b: str, gt: str, lookup: Dict[str, dict]
) -> Tuple[str, Set[str]]:
    """Return (error_in, codes) using the errored image's CV output."""
    ca, cb = lookup.get(path_a), lookup.get(path_b)
    codes_a = _parse_codes(ca) if ca else set()
    codes_b = _parse_codes(cb) if cb else set()
    score_a = (gt in codes_a, len(codes_a))
    score_b = (gt in codes_b, len(codes_b))
    if score_b >= score_a:
        return "B", codes_b
    return "A", codes_a


def export_study5_from_cv(
    lookup: Dict[str, dict], max_per_class: int = 100
) -> List[dict]:
    """Pairwise rows from CV batch (correct vs errored, same stem), up to max_per_class per type."""
    rows: List[dict] = []
    for key in ALL_ERROR_KEYS:
        for correct_p, errored_p in matched_correct_error_pairs(
            key, max_per_class=max_per_class, max_pairs=max_per_class
        ):
            img_a, img_b = _rel(correct_p), _rel(errored_p)
            if img_a not in lookup or img_b not in lookup:
                continue
            gt = key
            err_in, codes = _errored_side_codes(img_a, img_b, gt, lookup)
            pred = gt if gt in codes else (_primary(codes, gt) if codes else "")
            cv_b = lookup[img_b]
            elements = _scoped_elements(cv_b, gt, codes)
            rows.append({
                **_cv_row_base(),
                "image_a": img_a,
                "image_b": img_b,
                "error_side": "B",
                "ground_truth_error_type": gt,
                "error_in": err_in,
                "error_type": pred,
                "error_element": elements,
                "raw_response": json.dumps({
                    "pred_codes": sorted(codes),
                    "pred_elements": elements,
                }),
            })
    return rows


def export_study5(lookup: Dict[str, dict], template_rows: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for t in template_rows:
        img_a = _norm_image(t.get("image_a", ""))
        img_b = _norm_image(t.get("image_b", ""))
        gt = (t.get("ground_truth_error_type") or "").strip().upper()
        if not img_a or not img_b or gt not in ALL_ERROR_KEYS:
            continue
        err_in, codes = _errored_side_codes(img_a, img_b, gt, lookup)
        pred = gt if gt in codes else (_primary(codes, gt) if codes else "")
        cv_b = lookup.get(img_b, {})
        elements = _scoped_elements(cv_b, gt, codes)
        rows.append({
            **_cv_row_base(),
            "image_a": img_a,
            "image_b": img_b,
            "error_side": t.get("error_side", ""),
            "ground_truth_error_type": gt,
            "error_in": err_in,
            "error_type": pred,
            "error_element": elements,
            "raw_response": json.dumps({
                "pred_codes": sorted(codes),
                "pred_elements": elements,
            }),
        })
    if rows:
        return rows
    return export_study5_from_cv(lookup)


def _template(study: str) -> List[dict]:
    base = EVAL_RESULTS / RESULTS_SUBDIR if RESULTS_SUBDIR else EVAL_RESULTS
    p = base / REF_MODEL / study / "results.csv"
    return _read_csv_rows(p)


def _write_csv(study: str, rows: List[dict]) -> Path:
    out_dir = OUT_ROOT / study
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.csv"
    if not rows:
        print(f"  [warn] no rows for {study}")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {study}: {len(rows)} rows -> {path}")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export CV batch results to study CSVs")
    parser.add_argument(
        "--all-cv",
        action="store_true",
        help="Export every image in the CV batch (default). Do not intersect gemma3 templates.",
    )
    parser.add_argument(
        "--align-gemma3",
        action="store_true",
        help="Only export rows whose images appear in gemma3 study CSVs (smaller counts).",
    )
    parser.add_argument(
        "--max-pairs-per-class",
        type=int,
        default=100,
        help="Study 5: max correct/errored pairs per error type",
    )
    parser.add_argument(
        "--tier",
        choices=["easy", "med"],
        default="easy",
        help="Dataset tier: easy (default) or med",
    )
    parser.add_argument(
        "--cv-json",
        type=Path,
        default=None,
        help="CV batch results.json (default: tier-specific cv_shape_analyzer_v2[_med])",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Eval Results model folder for cv_v2 (default: cv_v2 or med/cv_v2)",
    )
    parser.add_argument(
        "--ref-model",
        default="gemma3",
        help="Template model folder when --align-gemma3 (default: gemma3)",
    )
    args = parser.parse_args()
    configure_tier(
        args.tier,
        cv_json=args.cv_json,
        out_root=args.out_root,
        ref_model=args.ref_model,
    )
    use_template = args.align_gemma3 and not args.all_cv

    print(f"Tier: {args.tier}")
    print(f"  CV JSON: {CV_ERR_JSON}")
    print(f"  Export:  {OUT_ROOT}")
    print(f"  Images:  {IMAGES_BASE_DIR}")
    if use_template:
        sub = RESULTS_SUBDIR or "(root)"
        print(f"  Templates: Eval Results/{sub}/{REF_MODEL}/")

    lookup = load_cv_lookup()
    print(f"CV lookup: {len(lookup)} images")
    if not lookup:
        raise SystemExit("No CV results — run run_shape_analyzer_dataset.py first.")

    def _tpl(study: str) -> List[dict]:
        return _template(study) if use_template else []

    exporters = {
        "study1_binary_classification": lambda: export_study1(lookup, _tpl("study1_binary_classification")),
        "study2_shape_error_detection": lambda: export_study2_or_3(
            lookup, _tpl("study2_shape_error_detection"), False
        ),
        "study3_connection_error_detection": lambda: export_study2_or_3(
            lookup, _tpl("study3_connection_error_detection"), True
        ),
        "study4_all_errors": lambda: export_study4(lookup, _tpl("study4_all_errors")),
        "study5_pairwise_detection": lambda: (
            export_study5(lookup, _tpl("study5_pairwise_detection"))
            if use_template
            else export_study5_from_cv(lookup, max_per_class=args.max_pairs_per_class)
        ),
    }
    for study in STUDY_DIRS:
        _write_csv(study, exporters[study]())
    print(f"\nDone. Notebook will pick up model folder: {OUT_ROOT}")


if __name__ == "__main__":
    main()

"""
Pair EasyOCR text blocks to CV-detected shape nodes (spatial overlap only).

Used to populate ``Elements`` for rule-based evaluation without using OCR for
error-type detection.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from erd_analyzer_shape_based_v2 import ERDError, ERDNode

from local_text_extractor import LocalTextExtractor

# Must match ERDAnalyzer.analyze() border padding.
ANALYZER_PAD = 20


def _bbox_contains_point(bbox: List[int], x: float, y: float, margin: int = 4) -> bool:
    x1, y1, x2, y2 = bbox
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def _distance_to_bbox(bbox: List[int], x: float, y: float) -> float:
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return float((dx * dx + dy * dy) ** 0.5)


def _normalize_label(text: str) -> str:
    """Collapse OCR noise; keep alphanumeric tokens for metadata overlap."""
    s = re.sub(r"\s+", " ", str(text).strip())
    return s


def assign_text_to_nodes(
    nodes: List["ERDNode"],
    text_blocks: List[dict],
    *,
    max_nearest_dist: float = 45.0,
) -> None:
    """
    Set ``node.label`` from OCR blocks whose center falls inside the node bbox.

    Falls back to the nearest node within ``max_nearest_dist`` pixels. Nodes
  without a match keep ``label`` empty (callers may fall back to ``node.id``).
    """
    for node in nodes:
        node.label = ""

    # (node, block) pairs inside bbox — prefer smallest node area per block.
    for block in text_blocks:
        cx, cy = block["center"]
        text = _normalize_label(block.get("text") or "")
        if not text:
            continue

        inside = [n for n in nodes if _bbox_contains_point(n.bbox, cx, cy)]
        if inside:
            target = min(
                inside,
                key=lambda n: (n.bbox[2] - n.bbox[0]) * (n.bbox[3] - n.bbox[1]),
            )
        else:
            dists = [(n, _distance_to_bbox(n.bbox, cx, cy)) for n in nodes]
            dists = [(n, d) for n, d in dists if d <= max_nearest_dist]
            if not dists:
                continue
            target = min(dists, key=lambda x: x[1])[0]

        if target.label:
            if text.lower() not in target.label.lower():
                target.label = f"{target.label} {text}"
        else:
            target.label = text


def _ocr_cache_path(image_path: Path, cache_dir: Optional[Path]) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    mtime = int(image_path.stat().st_mtime)
    return cache_dir / f"{image_path.stem}_{mtime}.json"


def pair_ocr_labels_to_nodes(
    image_path: Path,
    nodes: List["ERDNode"],
    *,
    language: str = "english",
    extractor: Optional[LocalTextExtractor] = None,
    pad: int = ANALYZER_PAD,
    cache_dir: Optional[Path] = None,
) -> List[dict]:
    """
    Run OCR on the padded diagram (same geometry as ``ERDAnalyzer``) and assign
    labels to ``nodes`` in place. Returns raw OCR blocks.
    """
    img = cv2.imread(str(image_path))
    if img is None or not nodes:
        return []

    padded = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    cache_file = _ocr_cache_path(image_path, cache_dir)
    blocks: List[dict] = []
    if cache_file is not None and cache_file.is_file():
        try:
            blocks = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            blocks = []

    if not blocks:
        ex = extractor or LocalTextExtractor()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / f"{image_path.stem}_padded.png"
            cv2.imwrite(str(tmp_path), padded)
            blocks = ex.extract_text_global(tmp_path, None, language)
        if cache_file is not None:
            cache_file.write_text(json.dumps(blocks), encoding="utf-8")

    assign_text_to_nodes(nodes, blocks)
    return blocks


def node_display_name(node: "ERDNode") -> str:
    """OCR label when present, otherwise internal node id."""
    lab = _normalize_label(getattr(node, "label", "") or "")
    return lab if lab else node.id


def elements_for_error_code(
    errors: List["ERDError"],
    nodes: List["ERDNode"],
    code: str,
) -> str:
    """
    Comma-separated OCR names for nodes involved in errors of ``code``.

    Shape errors use ``node_id``; connection errors use both endpoints of ``edge``.
    """
    if not code:
        return ""

    node_by_id = {n.id: n for n in nodes}
    labels: List[str] = []
    seen: set[str] = set()

    for err in errors:
        if err.code != code:
            continue
        node_ids: List[str] = []
        if err.node_id:
            node_ids = [err.node_id]
        elif err.edge:
            node_ids = list(err.edge)

        for nid in node_ids:
            node = node_by_id.get(nid)
            if not node:
                continue
            name = node_display_name(node)
            key = name.lower()
            if key and key not in seen:
                seen.add(key)
                labels.append(name)

    return ",".join(labels)


def elements_by_error_code(
    errors: List["ERDError"],
    nodes: List["ERDNode"],
    codes: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Map each error code present to its comma-separated element string."""
    from erd_analyzer_shape_based_v2 import ERROR_LABELS

    keys = codes if codes is not None else list(ERROR_LABELS.keys())
    return {code: elements_for_error_code(errors, nodes, code) for code in keys}

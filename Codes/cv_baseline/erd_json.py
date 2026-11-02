"""
Turn a Chen-style ER diagram image into the benchmark JSON schema.

The shape analyzer already finds rectangles, ovals, and diamonds. This module
uses a slightly wider stroke close so dashed (weak) shapes survive, reads
labels with the existing OCR pairing, and writes entities / relationships.
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from erd_analyzer_shape_based_v2 import ERDAnalyzer, ERDNode, SHAPE_TO_ROLE
from local_text_extractor import LocalTextExtractor
from text_node_pairing import assign_text_to_nodes

# Cardinality tokens drawn beside relationship edges, not inside a shape.
_CARD_RE = re.compile(
    r"^(?:1|n|m|0\.\.1|1\.\.1|0\.\.n|1\.\.n|0\.\.m|1\.\.m)$",
    re.IGNORECASE,
)
_ISA_RE = re.compile(r"^isa$", re.IGNORECASE)


def _classify_contour(contour, bw: int, bh: int) -> Optional[str]:
    """Label one closed contour as rectangle, oval, or diamond."""
    area = cv2.contourArea(contour)
    if bw < 36 or bh < 26 or area < 800:
        return None
    if bw / max(bh, 1) > 8 or bh / max(bw, 1) > 8:
        return None

    hull = cv2.convexHull(contour)
    extent = cv2.contourArea(hull) / max(bw * bh, 1)
    peri = cv2.arcLength(hull, True)
    if peri <= 0:
        return None
    approx_n = len(cv2.approxPolyDP(hull, 0.04 * peri, True))

    pts = hull.reshape(-1, 2)
    left, right = pts[np.argmin(pts[:, 0])], pts[np.argmax(pts[:, 0])]
    top, bottom = pts[np.argmin(pts[:, 1])], pts[np.argmax(pts[:, 1])]
    cx = (float(left[0]) + float(right[0])) / 2
    cy = (float(top[1]) + float(bottom[1])) / 2
    aligned = (
        abs(float(left[1]) - cy) / bh < 0.28
        and abs(float(right[1]) - cy) / bh < 0.28
        and abs(float(top[0]) - cx) / bw < 0.36
        and abs(float(bottom[0]) - cx) / bw < 0.36
    )

    # Graphviz rectangles fill their box; diamonds sit near 0.5; ovals near 0.78.
    if extent >= 0.88 and approx_n <= 6:
        return "rectangle"
    if aligned and 0.40 <= extent <= 0.68:
        return "diamond"
    if 0.70 <= extent <= 0.90:
        return "oval"
    return None


def _drop_containers(items: List[Tuple[str, np.ndarray, List[int]]]) -> List[Tuple[str, np.ndarray, List[int]]]:
    """Remove contours that swallow several smaller shapes (merged line blobs)."""
    kept = []
    for shape, contour, box in items:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        inside = 0
        for _, _, other in items:
            if other is box:
                continue
            oa = (other[2] - other[0]) * (other[3] - other[1])
            if oa >= area:
                continue
            cx = (other[0] + other[2]) / 2
            cy = (other[1] + other[3]) / 2
            if x1 + 6 <= cx <= x2 - 6 and y1 + 6 <= cy <= y2 - 6:
                inside += 1
        if inside >= 2:
            continue
        kept.append((shape, contour, box))
    return kept


def _nms(items: List[Tuple[str, np.ndarray, List[int]]]) -> List[Tuple[str, np.ndarray, List[int]]]:
    """Keep the larger contour when the stroke produces an inner and outer loop."""
    items = sorted(items, key=lambda it: (it[2][2] - it[2][0]) * (it[2][3] - it[2][1]), reverse=True)
    kept: List[Tuple[str, np.ndarray, List[int]]] = []
    for shape, contour, box in items:
        x1, y1, x2, y2 = box
        area = max((x2 - x1) * (y2 - y1), 1)
        duplicate = False
        for _, _, other in kept:
            ix1, iy1 = max(x1, other[0]), max(y1, other[1])
            ix2, iy2 = min(x2, other[2]), min(y2, other[3])
            if ix1 < ix2 and iy1 < iy2 and (ix2 - ix1) * (iy2 - iy1) > 0.45 * area:
                duplicate = True
                break
        if not duplicate:
            kept.append((shape, contour, box))
    return kept


def _detect_shapes(gray: np.ndarray) -> List[Tuple[ERDNode, np.ndarray]]:
    """Find ER shapes. A 9px dilation bridges dashed strokes without a square close."""
    _, inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(inv, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw: List[Tuple[str, np.ndarray, List[int]]] = []
    h, w = gray.shape[:2]
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh > 0.08 * h * w:
            continue
        shape = _classify_contour(contour, bw, bh)
        if shape not in SHAPE_TO_ROLE:
            continue
        raw.append((shape, contour, [x, y, x + bw, y + bh]))

    raw = _nms(_drop_containers(raw))
    raw.sort(key=lambda it: ((it[2][1] + it[2][3]) / 2, (it[2][0] + it[2][2]) / 2))

    nodes: List[Tuple[ERDNode, np.ndarray]] = []
    for index, (shape, contour, box) in enumerate(raw, start=1):
        x1, y1, x2, y2 = box
        nodes.append((
            ERDNode(
                id=f"n{index}",
                shape=shape,
                role=SHAPE_TO_ROLE[shape],
                bbox=box,
                center=[round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
            ),
            contour,
        ))
    return nodes


def _longest_dark_run(row: np.ndarray) -> int:
    best = current = 0
    for value in row:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _is_underlined(gray: np.ndarray, bbox: List[int]) -> bool:
    """True when a long horizontal stroke sits inside the shape (primary key)."""
    x1, y1, x2, y2 = bbox
    roi = gray[y1 + 3:y2 - 3, x1 + 5:x2 - 5]
    if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 12:
        return False
    dark = roi < 160
    height, width = dark.shape
    limit = 0.55 * width
    for row in range(int(height * 0.35), int(height * 0.85)):
        if _longest_dark_run(dark[row]) >= limit:
            return True
    return False


def _border_ink_ratio(gray: np.ndarray, contour: np.ndarray) -> float:
    """Fraction of the outline that sits on dark pixels.

    Detected contours are the inner hole of a dilated stroke, so the real ink
    lies a few pixels outside the hull.
    """
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(float)
    if len(hull) < 4:
        return 1.0
    center = hull.mean(axis=0)
    height, width = gray.shape[:2]
    dark = 0
    total = 0
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        length = max(int(np.linalg.norm(end - start)), 1)
        for step in range(0, length, 2):
            point = start + (end - start) * (step / length)
            vector = point - center
            norm = float(np.linalg.norm(vector)) + 1e-6
            outward = point + vector / norm * 7.0
            x, y = int(round(outward[0])), int(round(outward[1]))
            x0, x1 = max(0, x - 2), min(width, x + 3)
            y0, y1 = max(0, y - 2), min(height, y + 3)
            patch = gray[y0:y1, x0:x1]
            total += 1
            if patch.size and (patch < 180).any():
                dark += 1
    if total < 8:
        return 1.0
    return dark / total


def _is_dashed(gray: np.ndarray, contour: np.ndarray) -> bool:
    """True when the outline is broken into dashes (weak entity or relationship)."""
    return _border_ink_ratio(gray, contour) < 0.72


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _normalize_cardinality(text: str) -> str:
    token = re.sub(r"\s+", "", text).upper().replace("..", "..")
    token = token.replace("...", "..")
    if token in {"N", "M", "1"}:
        return token
    return token


def _clean_label(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _ocr_blocks(image_path: Path, extractor: LocalTextExtractor, pad: int) -> List[dict]:
    """Read text in the same padded coordinate frame as the detected shapes."""
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # extract_text_global reads from a path, so write the padded canvas once.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir) / f"{image_path.stem}_padded.png"
        cv2.imwrite(str(tmp), padded)
        return extractor.extract_text_global(tmp, None, "english")


def _glyph_tokens(
    gray: np.ndarray,
    nodes: Sequence[ERDNode],
    extractor: LocalTextExtractor,
    pairs: Sequence[Tuple[ERDNode, ERDNode]],
) -> List[Tuple[List[float], str]]:
    """Read 1/N/M marks that sit on edges. Full-image OCR skips these one-letter tokens."""
    _, inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=6)
    line_mask = np.zeros_like(inv)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
    ink = cv2.subtract(inv, line_mask)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)

    boxes = []
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        if not (3 <= bw <= 36 and 11 <= bh <= 28 and 20 <= area <= 450):
            continue
        cx, cy = float(centroids[index][0]), float(centroids[index][1])
        inside_shape = False
        for node in nodes:
            x1, y1, x2, y2 = node.bbox
            if x1 - 2 <= cx <= x2 + 2 and y1 - 2 <= cy <= y2 + 2:
                inside_shape = True
                break
        if inside_shape:
            continue
        near_edge = False
        for entity, rel in pairs:
            dist = _point_segment_distance(cx, cy, entity.center[0], entity.center[1], rel.center[0], rel.center[1])
            if dist <= 55:
                near_edge = True
                break
        if not near_edge:
            continue
        boxes.append((x, y, bw, bh, cx, cy))
    if not boxes:
        return []

    reader = extractor.get_reader("english")
    tokens: List[Tuple[List[float], str]] = []
    for x, y, bw, bh, cx, cy in boxes:
        crop = gray[max(0, y - 2): y + bh + 2, max(0, x - 2): x + bw + 2]
        if crop.size == 0:
            continue
        up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        up = cv2.copyMakeBorder(up, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
        reads = reader.readtext(up, allowlist="1NM")
        text = ""
        if reads:
            candidate, conf = reads[0][1], float(reads[0][2])
            candidate = candidate.strip().upper()
            if conf >= 0.6 and candidate in {"1", "N", "M"}:
                text = candidate
        if not text:
            text = "1" if bw <= 10 else "N"
        tokens.append(([cx, cy], text))
    return tokens


def _assign_cardinalities(
    nodes: Sequence[ERDNode],
    edges,
    blocks: List[dict],
    gray: np.ndarray,
    extractor: LocalTextExtractor,
) -> Dict[Tuple[str, str], str]:
    """Map each entity–relationship edge to the nearest 1/N/M label."""
    by_id = {node.id: node for node in nodes}
    pairs = []
    for edge in edges:
        a, b = by_id.get(edge.from_id), by_id.get(edge.to_id)
        if not a or not b:
            continue
        roles = {a.role, b.role}
        if roles == {"entity", "relationship"}:
            pairs.append((a, b) if a.role == "entity" else (b, a))

    tokens = _glyph_tokens(gray, nodes, extractor, pairs)
    for block in blocks:
        text = _clean_label(block.get("text") or "")
        if _CARD_RE.match(text.replace(" ", "")):
            tokens.append((block["center"], _normalize_cardinality(text)))

    assigned: Dict[Tuple[str, str], str] = {}
    used = set()
    for entity, rel in pairs:
        best_index = None
        best_dist = 70.0
        ax, ay = entity.center
        bx, by = rel.center
        for index, (center, _) in enumerate(tokens):
            if index in used:
                continue
            dist = _point_segment_distance(center[0], center[1], ax, ay, bx, by)
            if dist < best_dist:
                best_dist = dist
                best_index = index
        if best_index is not None:
            used.add(best_index)
            assigned[(entity.id, rel.id)] = tokens[best_index][1]
    return assigned


def _unique_name(name: str, used: Dict[str, int]) -> str:
    key = name or "unnamed"
    used[key] = used.get(key, 0) + 1
    if used[key] == 1:
        return key
    return f"{key}_{used[key]}"


def shapes_to_json(nodes: Sequence[ERDNode], edges, cardinalities: Dict[Tuple[str, str], str], flags: Dict[str, dict]) -> dict:
    """Build the benchmark entity/relationship object from a labeled shape graph."""
    by_id = {node.id: node for node in nodes}
    neighbors: Dict[str, List[str]] = {node.id: [] for node in nodes}
    for edge in edges:
        if edge.from_id in neighbors and edge.to_id in neighbors and edge.from_id != edge.to_id:
            neighbors[edge.from_id].append(edge.to_id)
            neighbors[edge.to_id].append(edge.from_id)

    entity_names: Dict[str, str] = {}
    used_entity_names: Dict[str, int] = {}
    for node in nodes:
        if node.role != "entity":
            continue
        label = _clean_label(node.label)
        if not label or _CARD_RE.match(label.replace(" ", "")):
            continue
        entity_names[node.id] = _unique_name(label, used_entity_names)

    entities: Dict[str, dict] = {}
    for node_id, name in entity_names.items():
        info = flags.get(node_id, {})
        entities[name] = {
            "attributes": [],
            "primary_keys": [],
            "weak": bool(info.get("dashed")),
        }

    # Attribute ovals hang off an entity. An oval hanging off another oval is composite.
    for node in nodes:
        if node.role != "attribute":
            continue
        label = _clean_label(node.label)
        if not label or _CARD_RE.match(label.replace(" ", "")):
            continue
        entity_ids = [nid for nid in neighbors[node.id] if nid in entity_names]
        parent_ovals = [
            nid for nid in neighbors[node.id]
            if by_id[nid].role == "attribute" and any(eid in entity_names for eid in neighbors[nid])
        ]
        info = flags.get(node.id, {})
        if entity_ids:
            target = entity_names[min(entity_ids, key=lambda eid: math.dist(node.center, by_id[eid].center))]
            bucket = entities[target]
            child_labels = []
            for nid in neighbors[node.id]:
                child = by_id[nid]
                if child.role != "attribute":
                    continue
                if any(eid in entity_names for eid in neighbors[nid]):
                    continue
                child_label = _clean_label(child.label)
                if child_label:
                    child_labels.append(child_label)
            if child_labels:
                bucket.setdefault("composite", {})[label] = child_labels
            elif info.get("dashed"):
                bucket.setdefault("derived", [])
                if label not in bucket["derived"]:
                    bucket["derived"].append(label)
            else:
                if label not in bucket["attributes"]:
                    bucket["attributes"].append(label)
                if info.get("underlined") and label not in bucket["primary_keys"]:
                    bucket["primary_keys"].append(label)
        elif parent_ovals:
            continue

    relationships = []
    for node in nodes:
        if node.role != "relationship":
            continue
        label = _clean_label(node.label)
        if not label:
            continue
        linked = []
        for nid in neighbors[node.id]:
            if nid in entity_names:
                linked.append(nid)
        # Stable left-to-right order so cardinality stays aligned with entities.
        linked.sort(key=lambda nid: (by_id[nid].center[0], by_id[nid].center[1]))
        if len(linked) < 2:
            continue
        info = flags.get(node.id, {})
        if _ISA_RE.match(label.replace(" ", "")):
            superclass = linked[0]
            relationships.append({
                "subclasses": [entity_names[nid] for nid in linked[1:]],
                "rel_name": "IsA",
                "superclass": entity_names[superclass],
            })
            continue
        relationships.append({
            "entities": [entity_names[nid] for nid in linked],
            "cardinality": [cardinalities.get((nid, node.id), "") for nid in linked],
            "name": label,
            "weak": bool(info.get("dashed")),
        })

    return {"entities": entities, "relationships": relationships}


def diagram_to_json(image_path: Path, extractor: Optional[LocalTextExtractor] = None, analyzer: Optional[ERDAnalyzer] = None) -> dict:
    """Extract shapes, read their labels, and return one benchmark JSON object."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        return {"entities": {}, "relationships": []}

    pad = 20
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    detected = _detect_shapes(gray)
    nodes = [node for node, _ in detected]

    flags = {}
    for node, contour in detected:
        flags[node.id] = {
            "dashed": _is_dashed(gray, contour),
            "underlined": node.role == "attribute" and _is_underlined(gray, node.bbox),
        }

    extractor = extractor or LocalTextExtractor()
    blocks = _ocr_blocks(image_path, extractor, pad)
    assign_text_to_nodes(nodes, blocks)

    analyzer = analyzer or ERDAnalyzer(use_clip=False)
    line_binary = analyzer._preprocess_for_connections(gray)
    edges = analyzer._detect_connections(line_binary, nodes, gray.shape[0], gray.shape[1])
    cardinalities = _assign_cardinalities(nodes, edges, blocks, gray, extractor)
    return shapes_to_json(nodes, edges, cardinalities, flags)

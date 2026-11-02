from __future__ import annotations

import cv2
import math
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any

def _load_shape_classifier():
    """Import CLIP shape classifier only when needed (avoids torch/PIL for --no-clip)."""
    try:
        from agents.shape_classifier import ShapeClassifier
    except Exception:
        from shape_classifier import ShapeClassifier
    return ShapeClassifier

ERROR_LABELS: Dict[str, str] = {
    "S1": "Entity as Attribute Shape",
    "S2": "Entity as Relationship Shape",
    "S3": "Relationship as Entity Shape",
    "S4": "Relationship as Attribute Shape",
    "S5": "Attribute as Relationship Shape",
    "S6": "Attribute as Entity Shape",
    "C1": "Attribute-to-Attribute Connection",
    "C2": "Relationship-to-Relationship Connection",
    "C3": "Entity-to-Entity Connection",
    "C4": "Attribute-to-Relationship Connection",
    "C5": "Entity-to-Attribute Connection",
}

SHAPE_TO_ROLE = {
    "rectangle": "entity",
    "diamond": "relationship",
    "oval": "attribute",
}

ROLE_TO_SHAPE = {
    "entity": "rectangle",
    "relationship": "diamond",
    "attribute": "oval",
}

# Direct connection errors from detected shapes.
CONNECTION_ERROR_CODE = {
    frozenset(["attribute", "attribute"]): "C1",
    frozenset(["relationship", "relationship"]): "C2",
    frozenset(["entity", "entity"]): "C3",
    frozenset(["attribute", "relationship"]): "C4",
    frozenset(["entity", "attribute"]): "C5",
}

# Under your current rule set, the only non-error pair is entity--relationship.
VALID_ROLE_PAIR = frozenset(["entity", "relationship"])

SHAPE_ERROR_CODE = {
    ("entity", "oval"): "S1",
    ("entity", "diamond"): "S2",
    ("relationship", "rectangle"): "S3",
    ("relationship", "oval"): "S4",
    ("attribute", "diamond"): "S5",
    ("attribute", "rectangle"): "S6",
}

@dataclass
class ERDNode:
    id: str
    shape: str
    role: str
    bbox: List[int]
    center: List[float]
    label: str = ""
    contour: Any = None

    def to_dict(self):
        d = asdict(self)
        d.pop("contour", None)
        return d

@dataclass
class ERDEdge:
    from_id: str
    to_id: str
    path: List[List[int]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

@dataclass
class ERDError:
    code: str
    label: str
    node_id: Optional[str] = None
    edge: Optional[Tuple[str, str]] = None
    details: str = ""

    def to_dict(self):
        return asdict(self)

@dataclass
class ERDResult:
    nodes: List[ERDNode]
    edges: List[ERDEdge]
    errors: List[ERDError]
    image_shape: Tuple[int, int]
    ok: bool = True
    issues: List[str] = field(default_factory=list)
    debug_image: Optional[np.ndarray] = None

    def to_dict(self):
        return {
            "ok": self.ok,
            "issues": self.issues,
            "image_shape": self.image_shape,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "errors": [e.to_dict() for e in self.errors],
        }

class ERDAnalyzer:
    """
    Shape-based ERD analyzer.

    No external labels are needed:
      rectangle -> entity
      diamond   -> relationship
      oval      -> attribute

    C errors are direct invalid shape-pair connections.
    S errors are inferred repair hints from invalid connections. Example:
      oval--diamond is C4, and the oval would need to be rectangle to make
      entity--relationship, so the oval is also flagged as S1.
    """

    MAX_SHAPE_AREA_RATIO = 0.50
    MIN_REAL_SHAPE_AREA = 200
    CONNECTION_MATCH_RADIUS = 70

    def __init__(self, use_clip: bool = True, infer_shape_errors: bool = True):
        self.use_clip = use_clip
        self.infer_shape_errors = infer_shape_errors
        self._shape_clf = _load_shape_classifier()() if use_clip else None

    def analyze(self, image_path: str | Path, save_debug_to: str | Path | None = None) -> ERDResult:
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            return ERDResult([], [], [], (0, 0), ok=False, issues=[f"Could not read image: {image_path}"])

        pad = 20
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.MIN_REAL_SHAPE_AREA = max(200, int(h * w * 0.0003))

        shape_binary = self._preprocess_for_shapes(gray)
        nodes = self._detect_shapes(shape_binary, h, w)

        line_binary = self._preprocess_for_connections(gray)
        edges = self._detect_connections(line_binary, nodes, h, w)

        errors = self._find_errors(nodes, edges)
        debug = self._draw_debug(img.copy(), nodes, edges, errors)
        if save_debug_to:
            cv2.imwrite(str(save_debug_to), debug)

        return ERDResult(nodes, edges, errors, (h, w), debug_image=debug)

    def _preprocess_for_shapes(self, gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(binary, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    def _preprocess_for_connections(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(edges, kernel, iterations=1)

    def _detect_shapes(self, binary: np.ndarray, h: int, w: int) -> List[ERDNode]:
        raw: List[ERDNode] = []
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_REAL_SHAPE_AREA or area > h * w * self.MAX_SHAPE_AREA_RATIO:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = bw / max(bh, 1)
            if aspect > 12.0 or aspect < 0.08:
                continue
            shape = self._classify_shape(contour, area, bw, bh)
            if shape not in SHAPE_TO_ROLE:
                continue
            raw.append(ERDNode(
                id="",
                shape=shape,
                role=SHAPE_TO_ROLE[shape],
                bbox=[x, y, x + bw, y + bh],
                center=[round(x + bw / 2, 1), round(y + bh / 2, 1)],
                contour=contour,
            ))
        return self._filter_overlapping_nodes(raw, h, w)

    def _filter_overlapping_nodes(self, nodes: List[ERDNode], h: int, w: int) -> List[ERDNode]:
        if not nodes:
            return []
        areas = [(n.bbox[2] - n.bbox[0]) * (n.bbox[3] - n.bbox[1]) for n in nodes]
        p85 = np.percentile(areas, 85)
        candidates = [n for n, a in zip(nodes, areas) if a < p85 * 5 and a < h * w * 0.4 and a >= self.MIN_REAL_SHAPE_AREA]

        def contains_many(c: ERDNode, others: List[ERDNode]) -> bool:
            x1, y1, x2, y2 = c.bbox
            bw, bh = x2 - x1, y2 - y1
            mx, my = max(bw * 0.08, 10), max(bh * 0.08, 10)
            count = sum(
                1 for o in others if o is not c
                and x1 + mx <= o.center[0] <= x2 - mx
                and y1 + my <= o.center[1] <= y2 - my
            )
            return count >= 3

        candidates = [n for n in candidates if not contains_many(n, candidates)]
        candidates.sort(key=lambda n: (n.bbox[2] - n.bbox[0]) * (n.bbox[3] - n.bbox[1]), reverse=True)

        keep: List[ERDNode] = []
        for n in candidates:
            n_area = (n.bbox[2] - n.bbox[0]) * (n.bbox[3] - n.bbox[1])
            duplicate = False
            for k in keep:
                if math.dist(n.center, k.center) < 20:
                    duplicate = True
                    break
                ix1, iy1 = max(n.bbox[0], k.bbox[0]), max(n.bbox[1], k.bbox[1])
                ix2, iy2 = min(n.bbox[2], k.bbox[2]), min(n.bbox[3], k.bbox[3])
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if inter > 0.50 * n_area:
                        duplicate = True
                        break
            if not duplicate:
                keep.append(n)

        keep.sort(key=lambda n: (n.center[1], n.center[0]))
        for i, n in enumerate(keep, start=1):
            n.id = f"n{i}"
        return keep

    def _classify_shape(self, contour, area: float, bw: int, bh: int) -> Optional[str]:
        """
        ERD-first shape classification.

        The old version was too eager to call rectangles because wide ovals have
        lower circularity. This version uses geometric evidence in this order:
          1) diamond/rhombus geometry,
          2) ellipse/oval geometry, including wide flat ovals,
          3) rectangle geometry only when corners and bbox-fill are strong,
          4) optional CLIP fallback for the remaining ambiguous cases.
        """
        if len(contour) < 5:
            return None

        if self._is_diamond_by_geometry(contour, bw, bh):
            return "diamond"

        # IMPORTANT: run oval before rectangle. Many ERD attributes are wide
        # horizontal ellipses whose circularity is not high enough for the old
        # 0.80 circularity rule, so they were falling through to rectangle.
        if self._is_oval_by_geometry(contour, area, bw, bh):
            return "oval"

        if self._is_rectangle_by_geometry(contour, bw, bh):
            return "rectangle"

        if self.use_clip and self._shape_clf is not None:
            label = self._shape_clf.classify(contour, area, bw, bh)
            # If CLIP says rectangle but geometry looks smooth/elliptical, trust
            # geometry. This fixes the common oval->rectangle failure mode.
            if label == "rectangle" and self._looks_smooth_like_oval(contour, bw, bh):
                return "oval"
            return label

        # With --no-clip, choose by geometry fallback instead of defaulting to
        # rectangle. This is crucial for your reported run: use_clip=false.
        return "oval" if self._looks_smooth_like_oval(contour, bw, bh) else "rectangle"

    def _corner_angle_stats(self, contour, eps_ratio: float = 0.035):
        """Return approx polygon and number of sharp-ish corners."""
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        if peri == 0:
            return None, 0
        approx = cv2.approxPolyDP(hull, eps_ratio * peri, True).reshape(-1, 2)
        if len(approx) < 3:
            return approx, 0

        sharp = 0
        for i in range(len(approx)):
            p0 = approx[i - 1].astype(float)
            p1 = approx[i].astype(float)
            p2 = approx[(i + 1) % len(approx)].astype(float)
            v1, v2 = p0 - p1, p2 - p1
            denom = max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-6)
            angle = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1, 1))))
            if angle < 135:
                sharp += 1
        return approx, sharp

    def _is_rectangle_by_geometry(self, contour, bw: int, bh: int) -> bool:
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        if peri == 0:
            return False

        hull_area = cv2.contourArea(hull)
        extent = hull_area / max(bw * bh, 1)
        approx, sharp = self._corner_angle_stats(contour, eps_ratio=0.025)
        if approx is None:
            return False

        # Rectangles fill their axis-aligned bbox strongly and have 4 corners.
        # Ovals may also have high extent for very flat shapes, so require sharp
        # corners; this prevents oval->rectangle.
        return len(approx) == 4 and sharp >= 4 and extent >= 0.82

    def _looks_smooth_like_oval(self, contour, bw: int, bh: int) -> bool:
        """Loose oval fallback for wide/flat ERD attributes."""
        if bw < 18 or bh < 18:
            return False

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False
        extent = hull_area / max(bw * bh, 1)

        # A filled ideal ellipse has extent pi/4 ~= 0.785. Edge/morphology can
        # push this around, so keep a broad band. Rectangles are usually >0.88;
        # diamonds are usually <0.65.
        approx_loose, sharp_loose = self._corner_angle_stats(contour, eps_ratio=0.025)
        approx_tight, sharp_tight = self._corner_angle_stats(contour, eps_ratio=0.012)
        n_loose = 0 if approx_loose is None else len(approx_loose)
        n_tight = 0 if approx_tight is None else len(approx_tight)

        smooth_polygon = (n_loose >= 6 or n_tight >= 8) and sharp_loose <= 2
        ellipse_extent = 0.62 <= extent <= 0.88

        # Ellipse fit sanity check: most contour points should lie close to the
        # fitted ellipse boundary. This helps separate ovals from rounded noise.
        fit_ok = False
        if len(contour) >= 5:
            try:
                (_, _), (MA, ma), _ = cv2.fitEllipse(contour)
                major, minor = max(MA, ma), min(MA, ma)
                if major > 0 and minor > 0:
                    fit_ok = (0.35 <= minor / major <= 1.0)
            except cv2.error:
                fit_ok = False

        return ellipse_extent and smooth_polygon and fit_ok

    def _is_oval_by_geometry(self, contour, area: float, bw: int, bh: int) -> bool:
        if area < 250 or bw < 18 or bh < 18:
            return False

        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return False

        circularity = (4 * math.pi * area) / (peri * peri)
        if circularity >= 0.78:
            return True

        # Wide/flat ovals have lower circularity, so do not reject them just
        # because circularity < 0.65. Use smoothness + ellipse extent instead.
        if self._looks_smooth_like_oval(contour, bw, bh):
            return True

        return False

    def _is_diamond_by_geometry(self, contour, bw: int, bh: int) -> bool:
        hull = cv2.convexHull(contour)
        if hull is None or len(hull) < 3:
            return False
        hull_area = cv2.contourArea(hull)
        peri = cv2.arcLength(hull, True)
        if peri == 0 or hull_area < 200:
            return False
        approx4 = None
        for eps in [0.04, 0.06, 0.08, 0.10, 0.12]:
            candidate = cv2.approxPolyDP(hull, eps * peri, True).reshape(-1, 2)
            if len(candidate) == 4:
                approx4 = candidate
                break
        if approx4 is None:
            return False
        extent = hull_area / max(bw * bh, 1)
        if extent > 0.75:
            return False
        cx, cy = np.mean(approx4, axis=0)
        pts = approx4[np.argsort(np.arctan2(approx4[:, 1] - cy, approx4[:, 0] - cx))]
        sides = [np.linalg.norm(pts[i].astype(float) - pts[(i + 1) % 4].astype(float)) for i in range(4)]
        if min(sides) / max(max(sides), 1) < 0.30:
            return False
        pts_by_x = pts[np.argsort(pts[:, 0])]
        pts_by_y = pts[np.argsort(pts[:, 1])]
        left_pt, right_pt = pts_by_x[0], pts_by_x[-1]
        top_pt, bottom_pt = pts_by_y[0], pts_by_y[-1]
        return (
            abs(left_pt[1] - cy) / max(bh, 1) < 0.35 and
            abs(right_pt[1] - cy) / max(bh, 1) < 0.35 and
            abs(top_pt[0] - cx) / max(bw, 1) < 0.35 and
            abs(bottom_pt[0] - cx) / max(bw, 1) < 0.35
        )

    def _detect_connections(self, binary: np.ndarray, nodes: List[ERDNode], h: int, w: int) -> List[ERDEdge]:
        mask = binary.copy()
        for n in nodes:
            x1, y1, x2, y2 = n.bbox
            pad = 8
            cv2.rectangle(mask, (max(0, x1 - pad), max(0, y1 - pad)), (min(w, x2 + pad), min(h, y2 + pad)), 0, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        edges: List[ERDEdge] = []
        seen = set()
        for cnt in contours:
            if cv2.contourArea(cnt) < 10 and cv2.arcLength(cnt, False) < 30:
                continue
            pts = cnt.reshape(-1, 2)
            if len(pts) < 2:
                continue
            p1, p2 = self._farthest_pair(pts)
            n1 = self._find_nearest_node(p1, nodes, self.CONNECTION_MATCH_RADIUS)
            n2 = self._find_nearest_node(p2, nodes, self.CONNECTION_MATCH_RADIUS)
            if not n1 or not n2 or n1.id == n2.id:
                continue
            pair = tuple(sorted([n1.id, n2.id]))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append(ERDEdge(pair[0], pair[1], [p1.tolist(), p2.tolist()]))
        return edges

    def _farthest_pair(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centroid = np.mean(pts, axis=0)
        p1 = pts[np.argmax(np.sum((pts - centroid) ** 2, axis=1))]
        p2 = pts[np.argmax(np.sum((pts - p1) ** 2, axis=1))]
        return p1, p2

    def _find_nearest_node(self, point: np.ndarray, nodes: List[ERDNode], max_dist: int) -> Optional[ERDNode]:
        px, py = float(point[0]), float(point[1])
        best, best_dist = None, float("inf")
        for n in nodes:
            x1, y1, x2, y2 = n.bbox
            if x1 <= px <= x2 and y1 <= py <= y2:
                dist = 0.0
            else:
                dist = math.sqrt(max(x1 - px, 0, px - x2) ** 2 + max(y1 - py, 0, py - y2) ** 2)
            if dist <= max_dist and dist < best_dist:
                best, best_dist = n, dist
        return best

    def _find_errors(self, nodes: List[ERDNode], edges: List[ERDEdge]) -> List[ERDError]:
        errors: List[ERDError] = []
        node_by_id = {n.id: n for n in nodes}
        seen_shape_errors = set()

        for edge in edges:
            a, b = node_by_id.get(edge.from_id), node_by_id.get(edge.to_id)
            if not a or not b:
                continue
            pair_roles = frozenset([a.role, b.role])
            c_code = CONNECTION_ERROR_CODE.get(pair_roles)
            if c_code:
                errors.append(ERDError(
                    code=c_code,
                    label=ERROR_LABELS[c_code],
                    edge=(edge.from_id, edge.to_id),
                    details=f"Detected invalid {a.role}-to-{b.role} connection from {a.shape}-to-{b.shape}.",
                ))

            if self.infer_shape_errors and c_code:
                for node, other in [(a, b), (b, a)]:
                    for intended_role in ["entity", "relationship", "attribute"]:
                        if intended_role == node.role:
                            continue
                        if frozenset([intended_role, other.role]) == VALID_ROLE_PAIR:
                            s_code = SHAPE_ERROR_CODE.get((intended_role, node.shape))
                            key = (node.id, s_code)
                            if s_code and key not in seen_shape_errors:
                                seen_shape_errors.add(key)
                                errors.append(ERDError(
                                    code=s_code,
                                    label=ERROR_LABELS[s_code],
                                    node_id=node.id,
                                    edge=(edge.from_id, edge.to_id),
                                    details=(
                                        f"Heuristic repair: node {node.id} is drawn as {node.shape}/{node.role}, "
                                        f"but changing its intended role to {intended_role} would make this edge valid."
                                    ),
                                ))
        return errors

    def _draw_debug(self, img: np.ndarray, nodes: List[ERDNode], edges: List[ERDEdge], errors: List[ERDError]) -> np.ndarray:
        # BGR: entity=red, attribute=green, relationship=orange; edges=purple
        shape_colors = {
            "rectangle": (0, 0, 255),
            "oval": (0, 255, 0),
            "diamond": (0, 140, 255),
        }
        edge_ok = (200, 80, 200)       # light purple
        edge_error = (255, 0, 180)   # vivid purple/magenta for invalid connections
        error_edges = {tuple(sorted(e.edge)) for e in errors if e.edge}

        for edge in edges:
            n1 = next((n for n in nodes if n.id == edge.from_id), None)
            n2 = next((n for n in nodes if n.id == edge.to_id), None)
            if not n1 or not n2:
                continue
            pair = tuple(sorted([edge.from_id, edge.to_id]))
            color = edge_error if pair in error_edges else edge_ok
            cv2.line(img, tuple(map(int, n1.center)), tuple(map(int, n2.center)), color, 2)

        for n in nodes:
            x1, y1, x2, y2 = n.bbox
            color = shape_colors.get(n.shape, (160, 160, 160))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{n.id}:{n.shape[:3]}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--debug", default="erd_debug.png")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--no-shape-error-inference", action="store_true")
    args = parser.parse_args()

    analyzer = ERDAnalyzer(
        use_clip=not args.no_clip,
        infer_shape_errors=not args.no_shape_error_inference,
    )
    result = analyzer.analyze(args.image, save_debug_to=args.debug)
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nDebug image saved to: {args.debug}")

if __name__ == "__main__":
    main()

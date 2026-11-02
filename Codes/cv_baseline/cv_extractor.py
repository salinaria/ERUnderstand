from __future__ import annotations

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
from agents.shape_classifier import ShapeClassifier


@dataclass
class CVNode:
    id: str
    shape: str
    bbox: List[int]
    center: List[float]
    contour: any = None

    def to_dict(self):
        return {"id": self.id, "shape": self.shape, "bbox": self.bbox, "center": self.center}

@dataclass
class CVEdge:
    from_id: str
    to_id: str
    label: str = ""
    label_en: str = ""
    arrow_path: List[List[int]] = field(default_factory=list)
    direction_confidence: float = 1.0

    def to_dict(self):
        return {
            "from": self.from_id, "to": self.to_id, "label": self.label,
            "label_en": self.label_en, "arrow_path": self.arrow_path,
            "direction_confidence": self.direction_confidence
        }

@dataclass
class CVExtractionResult:
    nodes: List[CVNode]
    edges: List[CVEdge]
    diagram_type: str
    image_shape: Tuple[int, int]
    debug_image: Optional[np.ndarray] = None
    ok: bool = True
    issues: List[str] = field(default_factory=list)


class CVExtractor:
    MAX_SHAPE_AREA_RATIO = 0.5
    ARROW_MATCH_RADIUS = 60
    # Minimum area is image-relative: ~0.03% of the image, floor at 200px.
    # This catches small terminals while ignoring letter-O / arrowhead noise.
    MIN_REAL_SHAPE_AREA = 200

    def __init__(self):
        from agents.shape_classifier import ShapeClassifier
        self._shape_clf = ShapeClassifier()

    def extract(self, image_path: Path, metrics=None) -> CVExtractionResult:
        img = cv2.imread(str(image_path))
        if img is None:
            return CVExtractionResult([], [], "unknown", (0, 0), ok=False, issues=["Could not read image"])

        # Pad the image so shapes at the very top/bottom/edges (Start, End terminals)
        # are not cut off by Canny / morphology boundary effects.
        PAD = 20
        img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Image-relative minimum area — grows with image size, floor at 200px
        self.MIN_REAL_SHAPE_AREA = max(200, int(h * w * 0.0003))

        # 1. Detect Shapes
        shape_binary = self._preprocess_for_shapes(gray)
        nodes = self._detect_shapes(shape_binary, h, w)

        # 2. Detect Arrows
        line_binary = self._preprocess_for_lines(gray)
        edges = self._detect_arrows(line_binary, nodes, h, w)

        # 3. Classify Diagram
        diagram_type = self._classify_diagram(nodes)

        # 4. Build Debug Image
        debug_img = self._draw_debug(img.copy(), nodes, edges)

        return CVExtractionResult(
            nodes=nodes, edges=edges, diagram_type=diagram_type,
            image_shape=(h, w), debug_image=debug_img, ok=True
        )

    def _preprocess_for_shapes(self, gray: np.ndarray) -> np.ndarray:
        """Aggressive preprocessing to stitch dashed lines together."""
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
        # Now run the same edge pipeline on the binarized image
        edges = cv2.Canny(binary, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return closed

    def _preprocess_for_lines(self, gray: np.ndarray) -> np.ndarray:
        """Light preprocessing to keep arrow lines thin and distinct."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(edges, kernel, iterations=1)

    def _detect_shapes(self, binary: np.ndarray, h: int, w: int) -> List[CVNode]:
        raw_nodes = []

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
            if shape is None:
                continue

            raw_nodes.append(CVNode(
                id="", shape=shape, bbox=[x, y, x + bw, y + bh],
                center=[round(x + bw / 2, 1), round(y + bh / 2, 1)], contour=contour
            ))

        return self._filter_overlapping_nodes(raw_nodes, h, w)

    def _filter_overlapping_nodes(self, nodes: List[CVNode], h: int, w: int) -> List[CVNode]:
        if not nodes: return []

        areas = [(n.bbox[2]-n.bbox[0])*(n.bbox[3]-n.bbox[1]) for n in nodes]
        p85_area = np.percentile(areas, 85)

        valid_size_nodes = []
        for n, area in zip(nodes, areas):
            if area < (p85_area * 5) and area < (h * w * 0.4):
                valid_size_nodes.append(n)

        valid_size_nodes = [
            n for n in valid_size_nodes
            if (n.bbox[2]-n.bbox[0])*(n.bbox[3]-n.bbox[1]) >= self.MIN_REAL_SHAPE_AREA
        ]

        # ── Ghost containment filter ──────────────────────────────────────────
        # A ghost is a large wrapper contour enclosing multiple real shapes.
        # Rule: if a node's bbox FULLY CONTAINS the centers of 3+ other nodes
        # (with inset margin so border-touching doesn't count), it's a ghost.
        # We use 3+ not 2+ because a large rectangle can legitimately have
        # 2 shapes just outside it whose centers are near its edge.
        def count_contained(candidate, others):
            x1, y1, x2, y2 = candidate.bbox
            bw = x2 - x1
            bh = y2 - y1
            # Inset margin: center must be well inside, not just on the border
            mx = max(bw * 0.08, 10)
            my = max(bh * 0.08, 10)
            return sum(
                1 for o in others
                if o is not candidate
                and (x1 + mx) <= o.center[0] <= (x2 - mx)
                and (y1 + my) <= o.center[1] <= (y2 - my)
            )

        valid_size_nodes = [
            n for n in valid_size_nodes
            if count_contained(n, valid_size_nodes) < 3
        ]

        valid_size_nodes = sorted(
            valid_size_nodes,
            key=lambda n: (n.bbox[2]-n.bbox[0])*(n.bbox[3]-n.bbox[1]),
            reverse=True
        )
        keep = []

        for n in valid_size_nodes:
            n_area = (n.bbox[2] - n.bbox[0]) * (n.bbox[3] - n.bbox[1])
            is_duplicate = False

            for k in keep:
                dist = math.sqrt((n.center[0]-k.center[0])**2 + (n.center[1]-k.center[1])**2)
                if dist < 20:
                    is_duplicate = True
                    break

                ix1, iy1 = max(n.bbox[0], k.bbox[0]), max(n.bbox[1], k.bbox[1])
                ix2, iy2 = min(n.bbox[2], k.bbox[2]), min(n.bbox[3], k.bbox[3])
                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area > 0.50 * n_area:
                        is_duplicate = True
                        break

            if not is_duplicate:
                keep.append(n)

        for idx, node in enumerate(keep):
            node.id = f"n{idx+1}"

        return keep

    def _is_diamond_by_geometry(self, contour, bw: int, bh: int) -> bool:
        """
        Geometric pre-screen for diamonds (rhombuses).

        ERD diamonds come in two flavours:
          A) "Square" diamond  — roughly equal width/height, rotated ~45°
             → minAreaRect angle ≈ 30–60°, low axis-aligned extent
          B) "Flat" diamond    — much wider than tall (e.g. 'generates', 'covers')
             → minAreaRect angle ≈ 0–20° BUT the 4-point convex hull has
               clearly concave top/bottom midpoints (the classic lens shape
               seen from above) AND low extent relative to bbox

        We work from the CONVEX HULL which survives morphological distortion
        far better than the raw contour.
        """
        hull = cv2.convexHull(contour)
        if hull is None or len(hull) < 3:
            return False

        hull_area = cv2.contourArea(hull)
        peri = cv2.arcLength(hull, True)
        if peri == 0 or hull_area < 200:
            return False

        # ── Check 1: Hull must approximate to 4 vertices ─────────────────────
        # Test a range of epsilons — a real diamond gives 4 consistently
        vertex_counts = []
        for eps in [0.04, 0.06, 0.08, 0.10, 0.12]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            vertex_counts.append(len(approx))

        has_4_vertices = any(v == 4 for v in vertex_counts)
        if not has_4_vertices:
            return False

        # ── Check 2: Low axis-aligned extent (diamonds leave empty corners) ───
        extent = hull_area / max(bw * bh, 1)
        if extent > 0.75:
            return False  # fills bbox too well → rectangle

        # ── Check 3: Get the best 4-point approximation ───────────────────────
        # Pick the epsilon that gives exactly 4 points
        approx4 = None
        for eps in [0.06, 0.08, 0.10, 0.04, 0.12]:
            candidate = cv2.approxPolyDP(hull, eps * peri, True).reshape(-1, 2)
            if len(candidate) == 4:
                approx4 = candidate
                break

        if approx4 is None:
            return False

        # ── Check 4: The 4 points must form a diamond/rhombus pattern ─────────
        # Sort points by angle from centroid → gives [left, top, right, bottom]
        # (or [top, right, bottom, left] depending on orientation)
        cx, cy = np.mean(approx4, axis=0)
        angles = np.arctan2(approx4[:, 1] - cy, approx4[:, 0] - cx)
        pts = approx4[np.argsort(angles)]  # ordered around the centroid

        # Diagonal 1: point 0 ↔ point 2  (opposite corners)
        # Diagonal 2: point 1 ↔ point 3  (opposite corners)
        d1 = np.linalg.norm(pts[0].astype(float) - pts[2].astype(float))
        d2 = np.linalg.norm(pts[1].astype(float) - pts[3].astype(float))

        if max(d1, d2) < 1:
            return False

        diag_ratio = min(d1, d2) / max(d1, d2)

        # For a "square" diamond d1 ≈ d2 (ratio → 1.0)
        # For a "flat" diamond  d1 >> d2  (ratio can be as low as 0.15)
        # We accept anything ≥ 0.12 — this rejects parallelograms which
        # have very unequal diagonals only when they're nearly degenerate
        if diag_ratio < 0.12:
            return False

        # ── Check 5: The centroid must be roughly central (not skewed) ────────
        # A parallelogram's centroid is central too, so we use one more check:
        # In a diamond, opposite sides are EQUAL LENGTH. Compute side lengths.
        side_lens = [
            np.linalg.norm(pts[i].astype(float) - pts[(i+1)%4].astype(float))
            for i in range(4)
        ]
        # Opposite sides should be similar
        opp1_ratio = min(side_lens[0], side_lens[2]) / max(max(side_lens[0], side_lens[2]), 1)
        opp2_ratio = min(side_lens[1], side_lens[3]) / max(max(side_lens[1], side_lens[3]), 1)
        if opp1_ratio < 0.5 or opp2_ratio < 0.5:
            return False  # sides too unequal → not a diamond

        # ── Check 6: Distinguish flat diamond from parallelogram ──────────────
        # A parallelogram has significant horizontal skew: the top two points
        # are offset from the bottom two points horizontally.
        # A flat diamond is symmetric: its leftmost and rightmost points are
        # at the vertical midline, and the top/bottom points are centered.
        #
        # Compute: sort by X → leftmost and rightmost points should be near
        # the vertical center (y ≈ cy) for a diamond.
        pts_by_x = pts[np.argsort(pts[:, 0])]
        left_pt  = pts_by_x[0]
        right_pt = pts_by_x[-1]

        # For a flat diamond the leftmost/rightmost points are near mid-height
        left_y_offset  = abs(left_pt[1]  - cy) / max(bh, 1)
        right_y_offset = abs(right_pt[1] - cy) / max(bh, 1)

        # Sort by Y → topmost and bottommost should be near horizontal center
        pts_by_y = pts[np.argsort(pts[:, 1])]
        top_pt    = pts_by_y[0]
        bottom_pt = pts_by_y[-1]
        top_x_offset    = abs(top_pt[0]    - cx) / max(bw, 1)
        bottom_x_offset = abs(bottom_pt[0] - cx) / max(bw, 1)

        # Diamond: left/right near vertical midline, top/bottom near horizontal midline
        # Parallelogram: top/bottom points are clearly offset from center
        is_symmetric = (
            left_y_offset  < 0.35 and
            right_y_offset < 0.35 and
            top_x_offset    < 0.35 and
            bottom_x_offset < 0.35
        )

        return is_symmetric

    def _is_oval_by_geometry(self, contour, area: float, bw: int, bh: int) -> bool:
        """
        Geometric oval pre-screen — runs before CLIP.
        Circularity = 4π·area/perimeter²: ellipses ~0.80-1.0, rects ~0.60-0.78, diamonds ~0.45-0.65.
        Also rejects shapes that are too small to be a real node (not an arrowhead or letter-O).
        """
        if area < 400 or bw < 18 or bh < 18:
            return False  # too small → arrowhead or letter artifact

        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return False
        circularity = (4 * math.pi * area) / (peri * peri)

        if circularity >= 0.80:
            return True   # Tier A: clearly oval

        if circularity < 0.65:
            return False  # fall through to CLIP

        # Tier B: moderate circularity — confirm with hull fill and vertex count
        hull = cv2.convexHull(contour)
        if hull is None or len(hull) < 3:
            return False
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False
        hull_peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.05 * hull_peri, True)
        return (area / hull_area) >= 0.88 and len(approx) > 6

    def _classify_shape(self, contour, area: float, bw: int, bh: int) -> Optional[str]:
        """
        Classification pipeline:
          1. Geometric diamond pre-screen
          2. Geometric oval pre-screen  ← catches ovals CLIP misses
          3. CLIP zero-shot for remaining ambiguous cases
        """
        if len(contour) < 5:
            return None

        if self._is_diamond_by_geometry(contour, bw, bh):
            return "diamond"

        if self._is_oval_by_geometry(contour, area, bw, bh):
            return "oval"

        label = self._shape_clf.classify(contour, area, bw, bh)
        if label == "parallelogram":
            label = "rectangle"
        return label

    def _detect_arrows(self, binary: np.ndarray, nodes: List[CVNode], h: int, w: int) -> List[CVEdge]:
        line_mask = binary.copy()

        for n in nodes:
            pad = 8
            cv2.rectangle(line_mask,
                          (max(0, n.bbox[0]-pad), max(0, n.bbox[1]-pad)),
                          (min(w, n.bbox[2]+pad), min(h, n.bbox[3]+pad)), 0, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edges = []
        seen = set()

        for cnt in contours:
            if cv2.contourArea(cnt) < 10 and cv2.arcLength(cnt, False) < 30:
                continue

            pts = cnt.reshape(-1, 2)
            if len(pts) < 2: continue

            centroid = np.mean(pts, axis=0)
            dists = np.sum((pts - centroid)**2, axis=1)
            p1 = pts[np.argmax(dists)]
            dists_p1 = np.sum((pts - p1)**2, axis=1)
            p2 = pts[np.argmax(dists_p1)]

            n1 = self._find_nearest_node(p1, nodes, self.ARROW_MATCH_RADIUS)
            n2 = self._find_nearest_node(p2, nodes, self.ARROW_MATCH_RADIUS)

            if n1 and n2 and n1.id != n2.id:
                pair = tuple(sorted([n1.id, n2.id]))
                if pair not in seen:
                    seen.add(pair)
                    n_from, n_to = self._determine_direction(p1, p2, n1, n2, binary)
                    edges.append(CVEdge(from_id=n_from.id, to_id=n_to.id, arrow_path=[p1.tolist(), p2.tolist()]))

        return edges

    def _find_nearest_node(self, point: np.ndarray, nodes: List[CVNode], max_dist: int) -> Optional[CVNode]:
        best_node, best_dist = None, float("inf")
        px, py = point[0], point[1]

        for node in nodes:
            x1, y1, x2, y2 = node.bbox
            if x1 <= px <= x2 and y1 <= py <= y2:
                d = 0
            else:
                d = math.sqrt(max(x1 - px, 0, px - x2)**2 + max(y1 - py, 0, py - y2)**2)
            if d < best_dist and d <= max_dist:
                best_dist, best_node = d, node
        return best_node

    def _determine_direction(self, p1, p2, n1: CVNode, n2: CVNode, binary: np.ndarray):
        h, w = binary.shape

        def region_density(pt, radius=15):
            px, py = int(pt[0]), int(pt[1])
            region = binary[max(0, py-radius):min(h, py+radius), max(0, px-radius):min(w, px+radius)]
            return np.sum(region > 0) / region.size if region.size > 0 else 0

        if region_density(p2) >= region_density(p1):
            return n1, n2
        return n2, n1

    def _classify_diagram(self, nodes: List[CVNode]) -> str:
        if not nodes: return "flowchart"
        shapes = [n.shape for n in nodes]
        if "diamond" in shapes or len(nodes) < 3: return "flowchart"
        if shapes.count("oval") > 2 and shapes.count("rectangle") > 0: return "erd"
        return "flowchart"

    def _draw_debug(self, img: np.ndarray, nodes: List[CVNode], edges: List[CVEdge]) -> np.ndarray:
        COLORS = {
            "rectangle":     (0, 255, 0),
            "diamond":       (0, 165, 255),
            "oval":          (255, 0, 0),
            "parallelogram": (255, 0, 255),
        }

        for node in nodes:
            x1, y1, x2, y2 = node.bbox
            color = COLORS.get(node.shape, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{node.id}:{node.shape[:3]}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for edge in edges:
            n1 = next((n for n in nodes if n.id == edge.from_id), None)
            n2 = next((n for n in nodes if n.id == edge.to_id), None)
            if n1 and n2:
                pt1 = (int(n1.center[0]), int(n1.center[1]))
                pt2 = (int(n2.center[0]), int(n2.center[1]))
                cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), 2, tipLength=0.1)

        return img

    def save_debug_image(self, result: CVExtractionResult, output_path: Path):
        if result.debug_image is not None:
            cv2.imwrite(str(output_path), result.debug_image)
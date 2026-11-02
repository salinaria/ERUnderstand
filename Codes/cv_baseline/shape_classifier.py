"""
shape_classifier.py
====================
Zero-shot shape classifier using OpenAI CLIP (ViT-B/32).
3 classes only: oval, diamond, rectangle.

The ShapeClassifier is a singleton — the model loads once per process
and is reused across all CVExtractor instances and all images.

INSTALLATION:
    pip install torch torchvision
    pip install git+https://github.com/openai/CLIP.git
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


RENDER_SIZE = 224

PROMPTS = {
    "oval": [
        "an oval shape",
        "an ellipse outline",
        "a rounded oval",
        "an egg shaped outline",
        "a circle or ellipse",
    ],
    "diamond": [
        "a diamond shape",
        "a rhombus rotated 45 degrees",
        "a square rotated 45 degrees",
        "a diamond outline with four pointed corners",
        "a flat wide diamond shape",
    ],
    "rectangle": [
        "a rectangle",
        "a rectangular box outline",
        "a square or rectangle",
        "a box shape with straight edges",
        "a parallelogram or slanted rectangle",
    ],
}


class _ShapeClassifierImpl:
    """Internal implementation — do not instantiate directly. Use ShapeClassifier."""

    def __init__(self):
        print("[ShapeClassifier] Loading CLIP ViT-B/32...")
        import clip
        import torch

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        self._text_features = {}
        with torch.no_grad():
            for label, prompts in PROMPTS.items():
                tokens = clip.tokenize(prompts).to(self.device)
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                self._text_features[label] = feats.mean(dim=0)

        print(f"[ShapeClassifier] Ready on {self.device}.")

    def _render_contour(self, contour: np.ndarray, bw: int, bh: int) -> Image.Image:
        pad = max(20, int(min(bw, bh) * 0.15))
        canvas = np.ones((bh + pad * 2, bw + pad * 2, 3), dtype=np.uint8) * 255
        shifted = (contour - contour.min(axis=0).reshape(1, 1, 2) + pad).astype(np.int32)
        cv2.drawContours(canvas, [shifted], -1, (160, 160, 160), -1)
        cv2.drawContours(canvas, [shifted], -1, (0, 0, 0), 2)
        return Image.fromarray(canvas).convert("RGB").resize(
            (RENDER_SIZE, RENDER_SIZE), Image.LANCZOS
        )

    def classify(self, contour, area: float, bw: int, bh: int) -> str:
        try:
            img = self._render_contour(contour, bw, bh)
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            with self._torch.no_grad():
                img_feat = self.model.encode_image(tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            scores = {
                label: (img_feat @ feat.unsqueeze(-1)).item()
                for label, feat in self._text_features.items()
            }
            return max(scores, key=scores.get)

        except Exception as e:
            print(f"[ShapeClassifier] Error: {e}")
            return "rectangle"


# ── Singleton instance — created once, shared across everything ───────────────
_instance: _ShapeClassifierImpl | None = None


def ShapeClassifier() -> _ShapeClassifierImpl:
    """
    Returns the global ShapeClassifier instance, creating it on first call.
    Subsequent calls return the same object — no reloading.

    Usage (unchanged from before):
        self._shape_clf = ShapeClassifier()
        label = self._shape_clf.classify(contour, area, bw, bh)
    """
    global _instance
    if _instance is None:
        _instance = _ShapeClassifierImpl()
    return _instance
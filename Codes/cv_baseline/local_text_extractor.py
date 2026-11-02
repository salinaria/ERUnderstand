from __future__ import annotations
import cv2
import numpy as np
import os
from pathlib import Path

# PIL for Arabic Unicode rendering (optional; English-only runs do not need these).
from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_RTL = True
except ImportError:
    HAS_RTL = False

try:
    import easyocr
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️ easyocr not installed. Run: pip install easyocr")

# EasyOCR uses different language codes
EASYOCR_LANG_MAP = {
    "english": ["en"], "french": ["fr", "en"], "german": ["de", "en"],
    "spanish": ["es", "en"], "arabic": ["ar", "en"], "persian": ["fa", "en"],
    "russian": ["ru", "en"], "romanian": ["ro", "en"]
}

class LocalTextExtractor:
    """Uses deep learning (EasyOCR) to read text regardless of background colors or shapes."""
    
    def __init__(self):
        self.readers = {}
        self._easyocr_gpu = self._resolve_easyocr_gpu()

    def _resolve_easyocr_gpu(self) -> bool:
        """
        Decide whether EasyOCR should run on CUDA.

        Priority:
        1) FORCE_EASYOCR_GPU=1 / true   -> always try GPU
        2) FORCE_EASYOCR_GPU=0 / false  -> force CPU
        3) default                       -> auto-detect via torch.cuda.is_available()
        """
        env_value = os.environ.get("FORCE_EASYOCR_GPU", "").strip().lower()
        if env_value in {"1", "true", "yes", "on"}:
            return True
        if env_value in {"0", "false", "no", "off"}:
            return False

        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def get_reader(self, language: str):
        langs = EASYOCR_LANG_MAP.get(language, ["en"])
        lang_key = tuple(langs)
        
        if lang_key not in self.readers:
            mode = "CUDA" if self._easyocr_gpu else "CPU"
            print(f"  [EasyOCR] Loading models for {langs} on {mode} (first time takes a minute)...")
            self.readers[lang_key] = easyocr.Reader(langs, gpu=self._easyocr_gpu)
            
        return self.readers[lang_key]

    def extract_text_global(
        self,
        image_path: Path,
        output_dir: Optional[Path],
        language: str,
        *,
        save_debug: bool = False,
    ) -> list:
        if not HAS_OCR:
            return []

        img_cv = cv2.imread(str(image_path))
        if img_cv is None:
            return []

        reader = self.get_reader(language)
        raw_results = reader.readtext(str(image_path))

        text_blocks = []
        draw = None
        img_pil = None
        font = None

        if save_debug and output_dir is not None:
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except IOError:
                font = ImageFont.load_default()

        for bbox, text, conf in raw_results:
            if conf > 0.10:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

                text_blocks.append({
                    "text": text,
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    "conf": conf,
                })

                if draw is not None and font is not None:
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                    display_text = text
                    if language in ["arabic", "persian"] and HAS_RTL:
                        display_text = get_display(arabic_reshaper.reshape(text))
                    text_bbox = draw.textbbox((0, 0), display_text, font=font)
                    tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                    draw.rectangle(
                        [x1, max(0, y1 - th - 5), x1 + tw, y1],
                        fill=(255, 255, 255),
                    )
                    draw.text((x1, max(0, y1 - th - 5)), display_text, font=font, fill=(255, 0, 0))

        if save_debug and output_dir is not None and img_pil is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            img_debug = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            debug_path = output_dir / f"{image_path.stem}_easyocr_raw.png"
            cv2.imwrite(str(debug_path), img_debug)

        return text_blocks
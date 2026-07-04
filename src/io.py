"""Чтение изображений (в т.ч. пути с кириллицей на Windows)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Панорамы до ~27000x21000 px
Image.MAX_IMAGE_PIXELS = max(getattr(Image, "MAX_IMAGE_PIXELS", 0) or 0, 600_000_000)


def imread(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось прочитать: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Не удалось записать: {path}")
    buf.tofile(str(path))


def get_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (width, height)


def imread_max_side(path: str | Path, max_side: int | None = None) -> tuple[np.ndarray, float]:
    """Читает изображение и при необходимости уменьшает длинную сторону."""
    img = imread(path)
    if max_side is None or max_side <= 0:
        return img, 1.0

    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img, 1.0

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def read_tile_bgr(
    path: str | Path,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Читает прямоугольный фрагмент без загрузки всей панорамы в память."""
    right = x + width
    bottom = y + height
    with Image.open(path) as im:
        crop = im.crop((x, y, right, bottom))
        rgb = np.array(crop.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

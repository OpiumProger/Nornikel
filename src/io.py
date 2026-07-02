"""Чтение изображений (в т.ч. пути с кириллицей на Windows)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


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

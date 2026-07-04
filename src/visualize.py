"""Визуализация: зелёный=обычные, красный=тонкие, синий=тальк."""

from __future__ import annotations

import cv2
import numpy as np

from .io import imwrite


def create_overlay(
    img_bgr: np.ndarray,
    ordinary_mask: np.ndarray,
    fine_mask: np.ndarray,
    talc_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = img_bgr.copy().astype(np.float32)

    colors = {
        "ordinary": np.array([0, 255, 0], dtype=np.float32),
        "fine": np.array([0, 0, 255], dtype=np.float32),
        "talc": np.array([255, 0, 0], dtype=np.float32),
    }

    # Порядок: обычные → тонкие → тальк сверху
    for mask, color in (
        (ordinary_mask, colors["ordinary"]),
        (fine_mask, colors["fine"]),
        (talc_mask, colors["talc"]),
    ):
        m = mask > 0
        if not np.any(m):
            continue
        overlay[m] = overlay[m] * (1 - alpha) + color * alpha

    return overlay.astype(np.uint8)


def save_overlay(path, img_bgr, ordinary, fine, talc) -> None:
    out = create_overlay(img_bgr, ordinary, fine, talc)
    imwrite(path, out)

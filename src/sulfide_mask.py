"""Детекция и классификация сульфидных срастаний."""

from __future__ import annotations

import cv2
import numpy as np

# Классы сульфидов в маске анализа
SULFIDE_ORDINARY = 1
SULFIDE_FINE = 2


def extract_sulfide_masks(
    img_bgr: np.ndarray,
    replacement_threshold: float = 0.22,
    min_area: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
  Возвращает:
    ordinary_mask, fine_mask, all_sulfide_mask  (uint8 0/255)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Сульфиды — светлые включения
    _, bright = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    ordinary = np.zeros_like(gray)
    fine = np.zeros_like(gray)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component = labels == i
        ys, xs = np.where(component)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        roi_gray = proc[y1 : y2 + 1, x1 : x2 + 1]
        roi_comp = component[y1 : y2 + 1, x1 : x2 + 1]

        # Доля тёмной фазы внутри светлого включения = степень замещения
        inside = roi_gray[roi_comp]
        if inside.size == 0:
            continue
        dark_ratio = np.mean(inside < np.percentile(inside, 40))

        target = fine if dark_ratio >= replacement_threshold else ordinary
        target[component] = 255

    all_sulfide = cv2.bitwise_or(ordinary, fine)
    return ordinary, fine, all_sulfide

"""Детекция и классификация сульфидных срастаний по ТЗ хакатона."""

from __future__ import annotations

import cv2
import numpy as np

SULFIDE_ORDINARY = 1
SULFIDE_FINE = 2


def extract_sulfide_masks(
    img_bgr: np.ndarray,
    min_area: int | None = None,
    ordinary_area_frac: float = 0.00008,
    replacement_threshold: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Обычные: крупные изолированные светлые сульфиды с минимальным замещением.
    Тонкие: мелкие или сильно замещённые нерудной (тёмной) фазой.

    Возвращает ordinary_mask, fine_mask, all_sulfide_mask (uint8 0/255).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    p90 = float(np.percentile(proc, 90))
    p85 = float(np.percentile(proc, 85))
    dark_ref = float(np.percentile(proc, 32))

    bright = (proc >= p85).astype(np.uint8) * 255
    blur = cv2.GaussianBlur(proc, (21, 21), 0)
    local_peak = proc.astype(np.float32) > blur.astype(np.float32) + 10
    bright = cv2.bitwise_and(bright, (local_peak.astype(np.uint8) * 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)

    if min_area is None:
        min_area = max(60, int(h * w * 0.000004))
    min_ordinary_area = max(350, int(h * w * ordinary_area_frac))

    ordinary = np.zeros_like(gray)
    fine = np.zeros_like(gray)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component = labels == i
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bbox_area = max(bw * bh, 1)
        compactness = area / bbox_area
        internal_mean = float(np.mean(proc[component]))

        comp_u8 = component.astype(np.uint8) * 255
        dilated = cv2.dilate(comp_u8, dilate_k, iterations=2)
        ring = (dilated > 0) & (~component)
        replacement = float(np.mean(proc[ring] < dark_ref)) if ring.any() else 0.0

        veins_inside = float(np.mean(proc[component] < dark_ref))

        is_ordinary = (
            area >= min_ordinary_area
            and compactness >= 0.30
            and internal_mean >= p90 - 8
            and replacement < replacement_threshold
            and veins_inside < 0.35
        )

        target = ordinary if is_ordinary else fine
        target[component] = 255

    all_sulfide = cv2.bitwise_or(ordinary, fine)
    return ordinary, fine, all_sulfide

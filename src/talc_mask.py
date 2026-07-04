"""Извлечение маски талька как тёмно-серой рассеянной фазы."""

from __future__ import annotations

import cv2
import numpy as np


def _blue_annotation_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Синие/голубые линии экспертной разметки -> заполненная маска."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Синий контур (как на ch1 «Области оталькования»)
    mask = cv2.inRange(hsv, np.array([95, 60, 40]), np.array([135, 255, 255]))
    # Циан
    mask |= cv2.inRange(hsv, np.array([80, 60, 40]), np.array([95, 255, 255]))

    # Утолщаем линии и заливаем области внутри контуров
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) < 30:
            continue
        cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

    return filled


def extract_annotation_mask(img_bgr: np.ndarray, min_pct: float = 0.05) -> tuple[np.ndarray, bool]:
    """Возвращает маску синей разметки и флаг, найдена ли она."""
    mask = _blue_annotation_mask(img_bgr)
    has_annotation = (np.count_nonzero(mask) / mask.size * 100) > min_pct
    return mask, has_annotation


def _remove_small_components(mask: np.ndarray, min_area: int, max_area: int | None = None) -> np.ndarray:
    cleaned = np.zeros_like(mask)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        cleaned[labels == i] = 255
    return cleaned


def extract_dark_talc_mask(
    img_bgr: np.ndarray,
    min_area: int | None = None,
    max_area_frac: float = 0.12,
) -> np.ndarray:
    """
    Детектор талька без синей разметки.

    Ищет тёмно-серую/слабо окрашенную фазу в нерудной матрице:
    - исключает яркие сульфиды;
    - исключает цветные линии разметки;
    - оставляет средне-тёмные матовые области, а не весь чёрный фон.
    """
    h, w = img_bgr.shape[:2]
    if min_area is None:
        min_area = max(80, int(h * w * 0.00001))
    max_area = max(min_area, int(h * w * max_area_frac))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    l_chan, a_chan, b_chan = cv2.split(lab)
    sat = hsv[:, :, 1]

    # Нормализуем освещение, но не превращаем яркие сульфиды в кандидаты талька.
    l_norm = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(l_chan)
    blur = cv2.GaussianBlur(l_norm, (7, 7), 0)

    p08, p18, p62, p88 = np.percentile(blur, [8, 18, 62, 88])
    sat_p80 = np.percentile(sat, 80)

    bright_sulfides = blur >= p88
    very_black_matrix = blur <= p08
    blue_markup = _blue_annotation_mask(img_bgr) > 0

    # Тальк в этих данных выглядит как средне-тёмная матовая фаза, а не как
    # самые чёрные провалы и не как яркие металлические сульфиды.
    chroma = cv2.absdiff(a_chan, 128) + cv2.absdiff(b_chan, 128)
    low_chroma = chroma <= np.percentile(chroma, 72)
    medium_dark = (blur >= p18) & (blur <= p62)
    weak_saturation = sat <= max(35, sat_p80)

    candidate = medium_dark & weak_saturation & low_chroma
    candidate &= ~bright_sulfides
    candidate &= ~very_black_matrix
    candidate &= ~blue_markup

    mask = candidate.astype(np.uint8) * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7, iterations=2)
    mask = _remove_small_components(mask, min_area=min_area, max_area=max_area)

    # Защита от "заливки" всей матрицы на тёмных панорамах.
    if np.count_nonzero(mask) / mask.size > 0.35:
        stricter = (blur >= np.percentile(blur, 25)) & (blur <= np.percentile(blur, 55))
        stricter &= weak_saturation & low_chroma & ~bright_sulfides & ~blue_markup
        mask = stricter.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7, iterations=2)
        mask = _remove_small_components(mask, min_area=min_area, max_area=max_area)

    return mask


def extract_talc_mask(img_bgr: np.ndarray, prefer_annotation: bool = True) -> tuple[np.ndarray, str]:
    """
    Возвращает (mask_uint8 0/255, method).
    method: 'dark_gray_phase' | 'none'
    """
    dark = extract_dark_talc_mask(img_bgr)
    dark_pct = np.count_nonzero(dark) / dark.size * 100

    if dark_pct > 0.2:
        return dark, "dark_gray_phase"

    return np.zeros(img_bgr.shape[:2], dtype=np.uint8), "none"

"""Извлечение маски талька: синие контуры разметки или тёмная рассеянная фаза."""

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


def _dark_scattered_mask(gray: np.ndarray) -> np.ndarray:
    """Тёмная рассеянная фаза в матрице (эвристика без разметки)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Адаптивный порог для тёмных включений
    dark = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    return dark


def extract_talc_mask(img_bgr: np.ndarray, prefer_annotation: bool = True) -> tuple[np.ndarray, str]:
    """
    Возвращает (mask_uint8 0/255, method).
    method: 'blue_annotation' | 'dark_scattered' | 'none'
    """
    ann = _blue_annotation_mask(img_bgr)
    ann_pct = np.count_nonzero(ann) / ann.size * 100

    if prefer_annotation and ann_pct > 0.1:
        return ann, "blue_annotation"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = _dark_scattered_mask(gray)
    dark_pct = np.count_nonzero(dark) / dark.size * 100

    if dark_pct > 1.0:
        return dark, "dark_scattered"

    return np.zeros_like(gray), "none"

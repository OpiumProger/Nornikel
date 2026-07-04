"""Предобработка OM-снимков."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE по L-каналу + лёгкое шумоподавление."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(out, d=5, sigmaColor=50, sigmaSpace=50)

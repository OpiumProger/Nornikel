"""Единый пайплайн извлечения маски талька: U-Net + тёмно-серая фаза."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.talc_mask import extract_dark_talc_mask, extract_talc_mask
from src.talc_unet import predict_talc_mask_unet


def extract_talc_mask_hybrid(
    img_bgr: np.ndarray,
    checkpoint_path: str | Path | None = None,
    threshold: float | str | None = "auto",
    use_unet: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, str, float | None]:
    """
    Возвращает:
      mask, prob_map, method, used_threshold
    """
    dark_mask = extract_dark_talc_mask(img_bgr)
    has_dark = bool(np.count_nonzero(dark_mask))
    prob_map = None
    used_threshold = None
    unet_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    if use_unet and checkpoint_path and Path(checkpoint_path).exists():
        try:
            unet_mask, prob_map, _, used_threshold = predict_talc_mask_unet(
                img_bgr,
                checkpoint_path=checkpoint_path,
                threshold=threshold,
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"U-Net талька недоступна: {exc}")
            use_unet = False

    if has_dark and use_unet and prob_map is not None:
        mask = np.maximum(unet_mask, dark_mask)
        return mask, prob_map, "unet+dark_gray_phase", used_threshold

    if has_dark:
        return dark_mask, None, "dark_gray_phase", None

    if use_unet and prob_map is not None:
        return unet_mask, prob_map, "unet", used_threshold

    mask, method = extract_talc_mask(img_bgr, prefer_annotation=False)
    return mask, None, method, None

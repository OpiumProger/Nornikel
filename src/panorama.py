"""Анализ панорамных OM-снимков (тайловый инференс + даунскейл)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.io import get_size, imread_max_side, resize_max_side
from src.metrics import classify_ore_result, compute_metrics, format_metrics_table
from src.ore_classifier import predict_ore_class_tiled
from src.preprocess import preprocess
from src.sulfide_mask import extract_sulfide_masks
from src.talc_mask import extract_talc_mask
from src.panorama_talc import predict_panorama_talc_tiled
from src.visualize import create_overlay

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PANORAMA_DIR = PROJECT_DIR / "data" / "raw" / "Панорамы"
DEFAULT_TALC_MODEL = PROJECT_DIR / "models" / "best_talc_unet.pth"
DEFAULT_CLASSIFIER_MODEL = PROJECT_DIR / "models" / "best_ore_resnet18.pth"


def list_panoramas(panorama_dir: Path | None = None) -> list[Path]:
    root = panorama_dir or DEFAULT_PANORAMA_DIR
    if not root.exists():
        return []

    def sort_key(path: Path) -> tuple[int, str]:
        return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)

    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        files.extend(root.glob(pattern))
    return sorted(set(files), key=sort_key)


def analyze_panorama(
    path: Path,
    *,
    max_side: int = 4096,
    preview_side: int = 2400,
    talc_model: Path | None = DEFAULT_TALC_MODEL,
    classifier_model: Path | None = DEFAULT_CLASSIFIER_MODEL,
    talc_threshold: float | str = "auto",
    use_unet: bool = True,
    use_classifier: bool = False,
    classifier_tile_size: int = 768,
    classifier_stride: int = 512,
    talc_mode: str = "tiled_stitch",
    talc_tile_size: int = 1536,
    talc_tile_stride: int = 1152,
    stitch_max_side: int = 8192,
) -> dict:
    """
    Анализ панорамы.

    talc_mode:
      - tiled_stitch: нарезка на тайлы с исходного разрешения + склейка U-Net (рекомендуется)
      - downscale: U-Net на уменьшенной копии (быстрее, хуже для мелкого талька)
    """
    from src.talc_pipeline import extract_talc_mask_hybrid

    orig_w, orig_h = get_size(path)
    img, scale = imread_max_side(path, max_side=max_side)
    proc = preprocess(img)

    prob_map = None
    talc_threshold_used = None
    talc_meta: dict = {"talc_mode": talc_mode}

    if use_unet and talc_model and talc_model.exists() and talc_mode == "tiled_stitch":
        try:
            talc_mask, prob_map, talc_method, talc_threshold_used, talc_meta = predict_panorama_talc_tiled(
                path,
                talc_model,
                tile_size=talc_tile_size,
                tile_stride=talc_tile_stride,
                stitch_max_side=stitch_max_side,
                threshold=talc_threshold,
            )
            if talc_mask.shape[:2] != proc.shape[:2]:
                talc_mask = cv2.resize(
                    talc_mask,
                    (proc.shape[1], proc.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                if prob_map is not None:
                    prob_map = cv2.resize(
                        prob_map,
                        (proc.shape[1], proc.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
        except Exception as exc:
            talc_mask, talc_method = extract_talc_mask(proc)
            talc_method = f"heuristic_fallback ({exc})"
    elif use_unet and talc_model and talc_model.exists():
        try:
            talc_mask, prob_map, talc_method, talc_threshold_used = extract_talc_mask_hybrid(
                img,
                checkpoint_path=talc_model,
                threshold=talc_threshold,
                use_unet=True,
            )
        except Exception as exc:
            talc_mask, talc_method = extract_talc_mask(proc)
            talc_method = f"heuristic_fallback ({exc})"
    else:
        talc_mask, talc_method = extract_talc_mask(proc)

    ordinary, fine, _ = extract_sulfide_masks(proc)
    metrics = compute_metrics(talc_mask, ordinary, fine, talc_method)

    classifier_result = None
    if use_classifier and classifier_model and classifier_model.exists():
        try:
            classifier_result = predict_ore_class_tiled(
                img,
                classifier_model,
                tile_size=classifier_tile_size,
                stride=classifier_stride,
            )
        except Exception as exc:
            classifier_result = {"error": str(exc)}
    result = classify_ore_result(metrics, classifier_result)

    overlay = create_overlay(proc, ordinary, fine, talc_mask)
    overlay_preview = resize_max_side(overlay, preview_side) if preview_side > 0 else overlay

    return {
        "input": str(path),
        "panorama": {
            "original_width": orig_w,
            "original_height": orig_h,
            "working_width": int(img.shape[1]),
            "working_height": int(img.shape[0]),
            "scale": round(scale, 6),
            "max_side": max_side,
            "megapixels_original": round(orig_w * orig_h / 1_000_000, 2),
            "megapixels_working": round(img.shape[0] * img.shape[1] / 1_000_000, 2),
            **talc_meta,
        },
        "result": result,
        "classifier": classifier_result,
        "talc_threshold_used": talc_threshold_used,
        "overlay": overlay,
        "overlay_preview": overlay_preview,
        "prob_map": prob_map,
        "masks": {"talc": talc_mask, "ordinary": ordinary, "fine": fine},
    }

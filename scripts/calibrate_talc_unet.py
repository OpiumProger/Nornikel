"""
Калибровка порога U-Net талька на валидационной выборке.

Запуск:
    python scripts/calibrate_talc_unet.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io import imread
from src.talc_unet import get_talc_predictor, postprocess_talc_mask
from train import load_pairs

PROJECT = Path(__file__).resolve().parent.parent
MODEL = PROJECT / "models" / "best_talc_unet.pth"
OUT_CONFIG = PROJECT / "models" / "talc_unet_config.json"


def dice(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
    pred = mask_pred > 0
    true = mask_true > 0
    inter = np.logical_and(pred, true).sum()
    union = pred.sum() + true.sum()
    if union == 0:
        return 1.0
    return float(2 * inter / union)


def _downscale_maps(
    prob: np.ndarray,
    gt: np.ndarray,
    max_side: int = 640,
) -> tuple[np.ndarray, np.ndarray, float]:
    import cv2

    h, w = prob.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return prob, gt, 1.0

    nh, nw = int(h * scale), int(w * scale)
    prob_s = cv2.resize(prob, (nw, nh), interpolation=cv2.INTER_LINEAR)
    gt_s = cv2.resize(gt, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return prob_s, gt_s, scale


def _score_threshold(
    cache_probs: list[np.ndarray],
    cache_masks: list[np.ndarray],
    thr: float,
    min_area_px: int,
    max_side: int,
) -> float:
    scores = []
    for prob, gt in zip(cache_probs, cache_masks):
        prob_s, gt_s, scale = _downscale_maps(prob, gt, max_side=max_side)
        area_thr = max(1, int(min_area_px * scale * scale))
        mask = (prob_s >= thr).astype(np.uint8) * 255
        mask = postprocess_talc_mask(mask, min_area_px=area_thr)
        scores.append(dice(mask, gt_s))
    return float(np.mean(scores))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число val-пар (0 = все)")
    parser.add_argument("--stride", type=int, default=0, help="Шаг тайлов (0 = из config)")
    parser.add_argument("--min-area-px", type=int, default=80)
    parser.add_argument("--max-side", type=int, default=640, help="Сторона для быстрого подбора порога")
    args = parser.parse_args()

    pairs = load_pairs()
    random.seed(42)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    if args.limit > 0:
        val_pairs = val_pairs[: args.limit]

    predictor = get_talc_predictor(MODEL)
    stride = args.stride or None
    thresholds = np.arange(0.03, 0.21, 0.01)
    best_thr = 0.04
    best_dice = -1.0
    min_area_px = args.min_area_px

    print(f"Калибровка на {len(val_pairs)} val-парах...", flush=True)
    cache_probs = []
    cache_masks = []

    for i, item in enumerate(val_pairs, 1):
        img = imread(item["image"])
        gt = imread(item["mask"])
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt = (gt > 127).astype(np.uint8) * 255
        prob = predictor.predict_probability_map(img, stride=stride)
        cache_probs.append(prob)
        cache_masks.append(gt)
        if i % 5 == 0 or i == len(val_pairs):
            print(f"  prob maps: {i}/{len(val_pairs)}", flush=True)

    print("Подбор порога...", flush=True)
    for i, thr in enumerate(thresholds, 1):
        mean_dice = _score_threshold(
            cache_probs,
            cache_masks,
            float(thr),
            min_area_px=min_area_px,
            max_side=args.max_side,
        )
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_thr = float(thr)
        if i % 6 == 0 or i == len(thresholds):
            print(f"  thresholds: {i}/{len(thresholds)} (best={best_thr:.2f}, dice={best_dice:.4f})", flush=True)

    config = {
        "threshold": round(best_thr, 3),
        "patch_size": predictor.patch_size,
        "tile_stride": int(predictor.config.get("tile_stride", 128)),
        "min_area_px": min_area_px,
        "val_dice_at_threshold": round(best_dice, 4),
        "val_pairs": len(val_pairs),
    }
    OUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Лучший порог: {best_thr:.3f}")
    print(f"Val dice:     {best_dice:.4f}")
    print(f"Сохранено:    {OUT_CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

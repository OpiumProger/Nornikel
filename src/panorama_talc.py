"""Тайловый U-Net-инференс талька на панорамах: нарезка → прогон → склейка."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.io import get_size, read_tile_bgr
from src.talc_mask import extract_dark_talc_mask
from src.talc_unet import (
    get_talc_predictor,
    postprocess_talc_mask,
    resolve_threshold,
)


def iter_tile_boxes(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
) -> list[tuple[int, int, int, int]]:
    """Возвращает (x, y, w, h) тайлов на исходной панораме."""
    if tile_size <= 0:
        raise ValueError("tile_size должен быть > 0")

    if width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]

    ys = list(range(0, max(height - tile_size + 1, 1), stride))
    xs = list(range(0, max(width - tile_size + 1, 1), stride))
    if ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    if xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))

    boxes: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            tw = min(tile_size, width - x)
            th = min(tile_size, height - y)
            boxes.append((x, y, tw, th))
    return boxes


def _stitch_dims(orig_w: int, orig_h: int, stitch_max_side: int) -> tuple[int, int, float]:
    scale = min(1.0, stitch_max_side / max(orig_w, orig_h))
    stitch_w = max(1, int(orig_w * scale))
    stitch_h = max(1, int(orig_h * scale))
    return stitch_w, stitch_h, scale


def _native_to_stitch_box(
    x: int,
    y: int,
    tw: int,
    th: int,
    orig_w: int,
    orig_h: int,
    stitch_w: int,
    stitch_h: int,
) -> tuple[int, int, int, int]:
    sx = int(round(x * stitch_w / orig_w))
    sy = int(round(y * stitch_h / orig_h))
    sw = max(1, int(round(tw * stitch_w / orig_w)))
    sh = max(1, int(round(th * stitch_h / orig_h)))
    sx = min(sx, max(stitch_w - 1, 0))
    sy = min(sy, max(stitch_h - 1, 0))
    sw = min(sw, stitch_w - sx)
    sh = min(sh, stitch_h - sy)
    return sx, sy, sw, sh


def _tile_blend_weight(height: int, width: int, margin_frac: float = 0.12) -> np.ndarray:
    """Плавные веса к центру тайла — убирает швы при склейке."""
    wy = np.ones(height, dtype=np.float32)
    wx = np.ones(width, dtype=np.float32)
    my = max(1, int(height * margin_frac))
    mx = max(1, int(width * margin_frac))
    ramp_y = np.linspace(0.15, 1.0, my, dtype=np.float32)
    ramp_x = np.linspace(0.15, 1.0, mx, dtype=np.float32)
    wy[:my] = ramp_y
    wy[-my:] = ramp_y[::-1]
    wx[:mx] = ramp_x
    wx[-mx:] = ramp_x[::-1]
    return np.outer(wy, wx)


def predict_panorama_talc_tiled(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    tile_size: int = 1536,
    tile_stride: int = 1152,
    stitch_max_side: int = 8192,
    threshold: float | str | None = "auto",
    device: str | None = None,
    config_path: str | Path | None = None,
    unet_inner_stride: int = 256,
    progress_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, str, float, dict]:
    """
    Нарезает панораму на тайлы (как в train-датасете), прогоняет U-Net, склеивает prob/mask.

    Возвращает:
      mask, prob_map, method, used_threshold, meta
    """
    path = Path(image_path)
    orig_w, orig_h = get_size(path)
    stitch_w, stitch_h, stitch_scale = _stitch_dims(orig_w, orig_h, stitch_max_side)

    predictor = get_talc_predictor(checkpoint_path, device=device, config_path=config_path)
    config = predictor.config

    prob_acc = np.zeros((stitch_h, stitch_w), dtype=np.float32)
    weight = np.zeros((stitch_h, stitch_w), dtype=np.float32)
    dark_acc = np.zeros((stitch_h, stitch_w), dtype=np.uint8)
    has_dark_phase = False

    boxes = iter_tile_boxes(orig_w, orig_h, tile_size, tile_stride)
    for i, (x, y, tw, th) in enumerate(boxes, 1):
        tile = read_tile_bgr(path, x, y, tw, th)
        prob_tile = predictor.predict_probability_map(tile, stride=unet_inner_stride)

        dark_tile = extract_dark_talc_mask(tile)
        dark_here = bool(np.count_nonzero(dark_tile))
        if dark_here:
            has_dark_phase = True

        sx, sy, sw, sh = _native_to_stitch_box(
            x, y, tw, th, orig_w, orig_h, stitch_w, stitch_h
        )
        prob_stitch = cv2.resize(prob_tile, (sw, sh), interpolation=cv2.INTER_LINEAR)
        blend = _tile_blend_weight(sh, sw)
        prob_acc[sy : sy + sh, sx : sx + sw] += prob_stitch * blend
        weight[sy : sy + sh, sx : sx + sw] += blend

        if dark_here:
            dark_stitch = cv2.resize(dark_tile, (sw, sh), interpolation=cv2.INTER_NEAREST)
            dark_acc[sy : sy + sh, sx : sx + sw] = np.maximum(
                dark_acc[sy : sy + sh, sx : sx + sw],
                dark_stitch,
            )

        if progress_every > 0 and (i % progress_every == 0 or i == len(boxes)):
            print(f"    U-Net тайлы: {i}/{len(boxes)}", flush=True)

    prob_map = prob_acc / np.maximum(weight, 1.0)

    area_scale = stitch_scale * stitch_scale
    min_area = max(1, int(round(int(config.get("min_area_px", 80)) * area_scale)))
    used_threshold = resolve_threshold(
        prob_map, threshold, config, min_area_px=min_area
    )
    unet_mask = (prob_map >= used_threshold).astype(np.uint8) * 255
    unet_mask = postprocess_talc_mask(unet_mask, min_area_px=min_area)

    if has_dark_phase:
        mask = np.maximum(unet_mask, dark_acc)
        method = "unet_tiled_stitch+dark_gray_phase"
    else:
        mask = unet_mask
        method = "unet_tiled_stitch"

    meta = {
        "talc_mode": "tiled_stitch",
        "tile_size": tile_size,
        "tile_stride": tile_stride,
        "stitch_max_side": stitch_max_side,
        "stitch_width": stitch_w,
        "stitch_height": stitch_h,
        "stitch_scale": round(stitch_scale, 6),
        "tiles_total": len(boxes),
        "min_area_px_stitch": min_area,
        "has_dark_phase": has_dark_phase,
    }
    return mask, prob_map, method, used_threshold, meta

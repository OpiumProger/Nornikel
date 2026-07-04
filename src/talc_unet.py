"""Инференс и постобработка U-Net для сегментации талька."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_DIR / "models" / "talc_unet_config.json"


def _build_model():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "Для U-Net инференса нужен segmentation_models_pytorch."
        ) from exc

    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def load_unet_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_CONFIG
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "threshold": 0.04,
        "patch_size": 256,
        "tile_stride": 128,
        "min_area_px": 80,
    }


def resolve_torch_device(torch, device: str | None = None):
    """Выбирает лучшее доступное устройство: CUDA, Apple MPS или CPU."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def postprocess_talc_mask(
    mask: np.ndarray,
    min_area_px: int = 120,
) -> np.ndarray:
    """Закрывает дыры и убирает мелкий шум (без агрессивного opening до фильтра площади)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if min_area_px <= 0:
        return mask

    cleaned = np.zeros_like(mask)
    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[labels == i] = 255
    return cleaned


def resolve_threshold(
    prob: np.ndarray,
    threshold: float | str | None,
    config: dict,
    min_area_px: int = 120,
) -> float:
    """Фиксированный, auto или калиброванный порог."""
    if threshold not in (None, "auto"):
        return float(threshold)

    calibrated = float(config.get("threshold", 0.12))
    if threshold != "auto":
        return calibrated

    # На панорамах auto-подбор по полной карте prob слишком медленный
    max_fit_pixels = 1_500_000
    if prob.size > max_fit_pixels:
        factor = (max_fit_pixels / prob.size) ** 0.5
        sw = max(64, int(prob.shape[1] * factor))
        sh = max(64, int(prob.shape[0] * factor))
        prob = cv2.resize(prob, (sw, sh), interpolation=cv2.INTER_AREA)
        min_area_px = max(1, int(min_area_px * factor * factor))

    p95 = float(np.percentile(prob, 95))
    p99 = float(np.percentile(prob, 99))
    adaptive = max(0.05, min(0.30, max(p95 * 0.85, p99 * 0.55)))

    candidates = [calibrated, adaptive, calibrated * 0.85, calibrated * 0.7, 0.08, 0.06, 0.05, 0.04, 0.03]
    for thr in sorted({round(c, 4) for c in candidates}, reverse=True):
        mask = (prob >= thr).astype(np.uint8) * 255
        cleaned = postprocess_talc_mask(mask, min_area_px=min_area_px)
        if cleaned.any():
            return float(thr)

    return float(min(adaptive, calibrated))


class TalcUNetPredictor:
    """Кэшируемый предиктор U-Net с тайловым инференсом."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        config_path: str | Path | None = None,
    ):
        import torch

        self.torch = torch
        self.device = resolve_torch_device(torch, device)
        self.config = load_unet_config(Path(config_path) if config_path else None)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.patch_size = int(checkpoint.get("patch_size", self.config.get("patch_size", 256)))
        self.model = _build_model().to(self.device)
        self.model.load_state_dict(checkpoint.get("model_state", checkpoint))
        self.model.eval()

    def _normalize(self, rgb: np.ndarray) -> np.ndarray:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = rgb.astype(np.float32) / 255.0
        return (rgb - mean) / std

    def _predict_patch(self, patch_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        rgb = self._normalize(rgb)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)
            return torch.sigmoid(logits)[0, 0].float().cpu().numpy()

    def predict_probability_map(
        self,
        img_bgr: np.ndarray,
        patch_size: int | None = None,
        stride: int | None = None,
    ) -> np.ndarray:
        """Тайловый инференс с усреднением перекрытий."""
        h, w = img_bgr.shape[:2]
        ps = patch_size or self.patch_size
        st = stride or int(self.config.get("tile_stride", max(ps // 2, 64)))

        if h <= ps and w <= ps:
            patch = cv2.resize(img_bgr, (ps, ps), interpolation=cv2.INTER_AREA)
            prob = self._predict_patch(patch)
            return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

        prob_acc = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        ys = list(range(0, max(h - ps + 1, 1), st))
        xs = list(range(0, max(w - ps + 1, 1), st))
        if ys[-1] != h - ps:
            ys.append(max(h - ps, 0))
        if xs[-1] != w - ps:
            xs.append(max(w - ps, 0))

        for y in ys:
            for x in xs:
                patch = img_bgr[y : y + ps, x : x + ps]
                if patch.shape[0] != ps or patch.shape[1] != ps:
                    patch = cv2.resize(patch, (ps, ps), interpolation=cv2.INTER_AREA)
                prob_patch = self._predict_patch(patch)
                prob_acc[y : y + ps, x : x + ps] += prob_patch
                weight[y : y + ps, x : x + ps] += 1.0

        weight = np.maximum(weight, 1.0)
        return prob_acc / weight

    def predict_mask(
        self,
        img_bgr: np.ndarray,
        threshold: float | str | None = "auto",
        patch_size: int | None = None,
        stride: int | None = None,
        min_area_px: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        prob = self.predict_probability_map(img_bgr, patch_size=patch_size, stride=stride)
        min_area = min_area_px if min_area_px is not None else int(self.config.get("min_area_px", 120))
        used_threshold = resolve_threshold(prob, threshold, self.config, min_area_px=min_area)
        mask = (prob >= used_threshold).astype(np.uint8) * 255
        mask = postprocess_talc_mask(mask, min_area_px=min_area)
        return mask, prob, used_threshold


_predictor_cache: dict[str, TalcUNetPredictor] = {}


def get_talc_predictor(
    checkpoint_path: str | Path,
    device: str | None = None,
    config_path: str | Path | None = None,
) -> TalcUNetPredictor:
    key = f"{checkpoint_path}|{device or 'auto'}|{config_path or 'default'}"
    if key not in _predictor_cache:
        _predictor_cache[key] = TalcUNetPredictor(checkpoint_path, device=device, config_path=config_path)
    return _predictor_cache[key]


def predict_talc_mask_unet(
    img_bgr: np.ndarray,
    checkpoint_path: str | Path,
    threshold: float | str | None = "auto",
    device: str | None = None,
    config_path: str | Path | None = None,
    max_side: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    predictor = get_talc_predictor(checkpoint_path, device=device, config_path=config_path)
    original_h, original_w = img_bgr.shape[:2]
    infer_img = img_bgr
    resized = False

    if max_side and max(original_h, original_w) > max_side:
        scale = max_side / max(original_h, original_w)
        infer_w = max(32, int(original_w * scale))
        infer_h = max(32, int(original_h * scale))
        infer_img = cv2.resize(img_bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        resized = True

    mask, prob, used_threshold = predictor.predict_mask(infer_img, threshold=threshold)
    if resized:
        prob = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255

    return mask, prob, f"unet:{predictor.device.type}", used_threshold

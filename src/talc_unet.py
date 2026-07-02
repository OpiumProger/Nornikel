"""Инференс обученной U-Net модели для сегментации талька."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _build_model():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "Для U-Net инференса нужен пакет segmentation_models_pytorch. "
            "Запусти в окружении new_chemberta_env."
        ) from exc

    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


class TalcUNetPredictor:
    """Лёгкая обёртка вокруг обученной модели талька."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _build_model().to(self.device)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _prepare(img_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        h, w = img_bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32
        padded = cv2.copyMakeBorder(
            resized,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_REFLECT_101,
        )
        return padded, (new_h, new_w), (h, w)

    def predict_mask(
        self,
        img_bgr: np.ndarray,
        threshold: float = 0.5,
        max_side: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает бинарную маску 0/255 и карту вероятностей 0..1."""
        torch = self.torch
        padded, resized_hw, original_hw = self._prepare(img_bgr, max_side=max_side)

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)
            prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

        new_h, new_w = resized_hw
        prob = prob[:new_h, :new_w]

        orig_h, orig_w = original_hw
        prob_full = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask = (prob_full >= threshold).astype(np.uint8) * 255

        # Небольшая чистка шума для более понятного оверлея.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask, prob_full


def predict_talc_mask_unet(
    img_bgr: np.ndarray,
    checkpoint_path: str | Path,
    threshold: float = 0.5,
    max_side: int = 1024,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    predictor = TalcUNetPredictor(checkpoint_path, device=device)
    mask, prob = predictor.predict_mask(img_bgr, threshold=threshold, max_side=max_side)
    return mask, prob, "unet"

"""Инференс ResNet18-классификатора типа руды."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from train_classifier import ID_TO_LABEL, LABEL_RU, ResNet18, center_crop, imread_rgb, normalize, resize_short_side


def resolve_torch_device(device: str | None = None) -> torch.device:
    """Выбирает лучшее доступное устройство: CUDA, Apple MPS или CPU."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


class OreClassifierPredictor:
    """Кэшируемый предиктор ResNet18."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_torch_device(device)
        checkpoint = _load_checkpoint(self.checkpoint_path, self.device)

        id_to_label = checkpoint.get("id_to_label", ID_TO_LABEL)
        self.id_to_label = {int(k): v for k, v in id_to_label.items()}
        self.image_size = int(checkpoint.get("image_size", 224))

        self.model = ResNet18(num_classes=len(self.id_to_label)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict_from_rgb(self, img_rgb: np.ndarray) -> dict:
        img = center_crop(resize_short_side(img_rgb, self.image_size), self.image_size)
        x = normalize(img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_id = int(np.argmax(probs))
        pred_label = self.id_to_label[pred_id]

        return {
            "pred_label": pred_label,
            "pred_label_ru": LABEL_RU.get(pred_label, pred_label),
            "confidence": float(probs[pred_id]),
            "probabilities": {
                self.id_to_label[i]: float(probs[i]) for i in range(len(probs))
            },
            "model_path": str(self.checkpoint_path),
            "device": self.device.type,
            "tiles_used": 1,
        }

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        img = imread_rgb(image_path)
        return self.predict_from_rgb(img)

    @torch.no_grad()
    def predict_tiled(
        self,
        img_bgr: np.ndarray,
        tile_size: int = 768,
        stride: int = 512,
        batch_size: int = 8,
    ) -> dict:
        import cv2

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        ps = self.image_size

        if max(h, w) <= int(ps * 1.5):
            return self.predict_from_rgb(rgb)

        ys = list(range(0, max(h - tile_size + 1, 1), stride))
        xs = list(range(0, max(w - tile_size + 1, 1), stride))
        if ys[-1] != h - tile_size:
            ys.append(max(h - tile_size, 0))
        if xs[-1] != w - tile_size:
            xs.append(max(w - tile_size, 0))

        tensors = []
        for y in ys:
            for x in xs:
                patch = rgb[y : y + tile_size, x : x + tile_size]
                patch = center_crop(resize_short_side(patch, ps), ps)
                tensors.append(normalize(patch))

        prob_acc = np.zeros(len(self.id_to_label), dtype=np.float64)
        tiles_used = 0
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(self.device)
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            prob_acc += probs.sum(axis=0)
            tiles_used += len(probs)

        probs = prob_acc / max(tiles_used, 1)
        pred_id = int(np.argmax(probs))
        pred_label = self.id_to_label[pred_id]

        return {
            "pred_label": pred_label,
            "pred_label_ru": LABEL_RU.get(pred_label, pred_label),
            "confidence": float(probs[pred_id]),
            "probabilities": {
                self.id_to_label[i]: float(probs[i]) for i in range(len(probs))
            },
            "model_path": str(self.checkpoint_path),
            "device": self.device.type,
            "tiles_used": tiles_used,
        }


_predictor_cache: dict[str, OreClassifierPredictor] = {}


def predict_ore_class(
    image_path: str | Path,
    checkpoint_path: str | Path,
    device: str | None = None,
) -> dict:
    key = f"{checkpoint_path}|{device or 'auto'}"
    if key not in _predictor_cache:
        _predictor_cache[key] = OreClassifierPredictor(checkpoint_path, device=device)
    return _predictor_cache[key].predict(image_path)


def predict_ore_class_tiled(
    img_bgr: np.ndarray,
    checkpoint_path: str | Path,
    device: str | None = None,
    tile_size: int = 768,
    stride: int = 512,
    batch_size: int = 8,
) -> dict:
    key = f"{checkpoint_path}|{device or 'auto'}"
    if key not in _predictor_cache:
        _predictor_cache[key] = OreClassifierPredictor(checkpoint_path, device=device)
    return _predictor_cache[key].predict_tiled(
        img_bgr,
        tile_size=tile_size,
        stride=stride,
        batch_size=batch_size,
    )

"""Инференс ResNet18-классификатора руды на одном изображении."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from train_classifier import ID_TO_LABEL, LABEL_RU, ResNet18, center_crop, imread_rgb, normalize, resize_short_side

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


def load_classifier(checkpoint_path: Path, device: torch.device) -> tuple[ResNet18, int, dict[int, str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    id_to_label = checkpoint.get("id_to_label", ID_TO_LABEL)
    # JSON/checkpoint keys can become strings.
    id_to_label = {int(k): v for k, v in id_to_label.items()}
    image_size = int(checkpoint.get("image_size", 224))

    model = ResNet18(num_classes=len(id_to_label)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, image_size, id_to_label


@torch.no_grad()
def classify_image(
    image_path: Path,
    model_path: Path,
    device_name: str | None = None,
) -> dict:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, image_size, id_to_label = load_classifier(model_path, device)

    img = imread_rgb(image_path)
    img = center_crop(resize_short_side(img, image_size), image_size)
    x = normalize(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs))
    pred_label = id_to_label[pred_id]

    return {
        "file": str(image_path),
        "pred_label": pred_label,
        "pred_label_ru": LABEL_RU.get(pred_label, pred_label),
        "confidence": float(probs[pred_id]),
        "probabilities": {
            id_to_label[i]: float(probs[i]) for i in range(len(probs))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Классификация типа руды по одному снимку")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "best_ore_resnet18.pth",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Файл не найден: {args.image}")
        return 1
    if not args.model.exists():
        print(f"Модель не найдена: {args.model}")
        return 1

    result = classify_image(args.image, args.model, device_name=args.device)
    print(
        f"Класс: {result['pred_label_ru']} "
        f"({result['pred_label']}), confidence={result['confidence']:.3f}"
    )
    for label, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {prob:.3f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

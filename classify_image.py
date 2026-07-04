"""Инференс ResNet18-классификатора руды на одном изображении."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ore_classifier import predict_ore_class

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


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

    result = predict_ore_class(args.image, args.model, device=args.device)
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

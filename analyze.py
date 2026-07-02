"""
End-to-end анализ одного OM-снимка.

Использование:
    python analyze.py path/to/image.jpg
    python analyze.py path/to/image.jpg --output reports/run1
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.io import imread, imwrite
from src.metrics import classify_ore, compute_metrics
from src.preprocess import preprocess
from src.sulfide_mask import extract_sulfide_masks
from src.talc_mask import extract_talc_mask
from src.talc_unet import predict_talc_mask_unet
from src.visualize import create_overlay

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


def analyze_image(
    path: Path,
    talc_model: Path | None = None,
    talc_threshold: float = 0.5,
    unet_max_side: int = 1024,
    use_unet: bool = True,
) -> dict:
    img = imread(path)
    proc = preprocess(img)

    prob_map = None
    if use_unet and talc_model and talc_model.exists():
        try:
            talc_mask, prob_map, talc_method = predict_talc_mask_unet(
                img,
                checkpoint_path=talc_model,
                threshold=talc_threshold,
                max_side=unet_max_side,
            )
        except Exception as exc:
            print(f"[warn] U-Net талька недоступна, fallback на эвристику: {exc}")
            talc_mask, talc_method = extract_talc_mask(proc)
    else:
        talc_mask, talc_method = extract_talc_mask(proc)

    ordinary, fine, _ = extract_sulfide_masks(proc)
    metrics = compute_metrics(talc_mask, ordinary, fine, talc_method)
    result = classify_ore(metrics)
    overlay = create_overlay(proc, ordinary, fine, talc_mask)

    return {
        "input": str(path),
        "result": result,
        "overlay": overlay,
        "prob_map": prob_map,
        "masks": {"talc": talc_mask, "ordinary": ordinary, "fine": fine},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Анализ OM-снимка руды")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument(
        "--talc-model",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "best_talc_unet.pth",
        help="Путь к обученной U-Net модели талька",
    )
    parser.add_argument("--no-unet", action="store_true", help="Не использовать U-Net, только эвристику")
    parser.add_argument("--talc-threshold", type=float, default=0.5)
    parser.add_argument("--unet-max-side", type=int, default=1024)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Файл не найден: {args.image}")
        return 1

    out = analyze_image(
        args.image,
        talc_model=args.talc_model,
        talc_threshold=args.talc_threshold,
        unet_max_side=args.unet_max_side,
        use_unet=not args.no_unet,
    )
    result = out["result"]
    m = result.metrics

    print(result.summary)
    print(f"  Сульфиды всего: {m.sulfide_pct}%")
    print(f"  Обычные: {m.ordinary_pct}% | Тонкие: {m.fine_pct}%")
    print(f"  Тальк ({m.talc_method}): {m.talc_pct}%")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        stem = args.image.stem
        imwrite(args.output / f"{stem}_overlay.jpg", out["overlay"])
        imwrite(args.output / f"{stem}_talc.png", out["masks"]["talc"])
        imwrite(args.output / f"{stem}_ordinary.png", out["masks"]["ordinary"])
        imwrite(args.output / f"{stem}_fine.png", out["masks"]["fine"])
        if out["prob_map"] is not None:
            prob = (out["prob_map"] * 255).clip(0, 255).astype("uint8")
            imwrite(args.output / f"{stem}_talc_probability.png", prob)

        report = {
            "file": args.image.name,
            "ore_class": result.ore_class,
            "ore_class_ru": result.ore_class_ru,
            "summary": result.summary,
            "talc_model": str(args.talc_model) if not args.no_unet else None,
            "metrics": m.__dict__,
        }
        with open(args.output / f"{stem}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\nСохранено в {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

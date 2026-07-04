"""
End-to-end анализ одного OM-снимка.

Использование:
    python analyze.py path/to/image.jpg
    python analyze.py path/to/image.jpg --output reports/run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.io import imread, imwrite
from src.metrics import classify_ore_result, compute_metrics, format_classifier_note, format_metrics_table
from src.ore_classifier import predict_ore_class
from src.preprocess import preprocess
from src.sulfide_mask import extract_sulfide_masks
from src.talc_mask import extract_talc_mask
from src.talc_pipeline import extract_talc_mask_hybrid
from src.visualize import create_overlay

DEFAULT_TALC_MODEL = Path(__file__).resolve().parent / "models" / "best_talc_unet.pth"
DEFAULT_CLASSIFIER_MODEL = Path(__file__).resolve().parent / "models" / "best_ore_resnet18.pth"


def parse_talc_threshold(value: str) -> float | str:
    if value.lower() == "auto":
        return "auto"
    return float(value)


def analyze_image(
    path: Path,
    talc_model: Path | None = None,
    classifier_model: Path | None = None,
    talc_threshold: float | str = "auto",
    talc_heuristic: str = "legacy",
    unet_max_side: int | None = 1536,
    use_unet: bool = True,
    use_classifier: bool = False,
) -> dict:
    img = imread(path)
    proc = preprocess(img)

    prob_map = None
    talc_threshold_used = None

    if use_unet and talc_model and talc_model.exists():
        try:
            talc_mask, prob_map, talc_method, talc_threshold_used = extract_talc_mask_hybrid(
                img,
                checkpoint_path=talc_model,
                threshold=talc_threshold,
                use_unet=True,
                heuristic=talc_heuristic,
                max_side=unet_max_side,
            )
        except Exception as exc:
            print(f"[warn] U-Net талька недоступна, fallback на эвристику: {exc}")
            talc_mask, talc_method = extract_talc_mask(img, heuristic=talc_heuristic)
    else:
        talc_mask, talc_method = extract_talc_mask(img, heuristic=talc_heuristic)

    ordinary, fine, _ = extract_sulfide_masks(proc)
    metrics = compute_metrics(talc_mask, ordinary, fine, talc_method)

    classifier_result = None
    if use_classifier and classifier_model and classifier_model.exists():
        try:
            classifier_result = predict_ore_class(path, classifier_model)
        except Exception as exc:
            print(f"[warn] ResNet-классификатор недоступен: {exc}")
    result = classify_ore_result(metrics, classifier_result)

    overlay = create_overlay(proc, ordinary, fine, talc_mask)

    return {
        "input": str(path),
        "result": result,
        "overlay": overlay,
        "prob_map": prob_map,
        "classifier": classifier_result,
        "talc_threshold_used": talc_threshold_used,
        "masks": {"talc": talc_mask, "ordinary": ordinary, "fine": fine},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Анализ OM-снимка руды")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument(
        "--talc-model",
        type=Path,
        default=DEFAULT_TALC_MODEL,
        help="Путь к обученной U-Net модели талька",
    )
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=DEFAULT_CLASSIFIER_MODEL,
        help="Путь к ResNet-классификатору руды",
    )
    parser.add_argument("--no-unet", action="store_true", help="Не использовать U-Net, только эвристику талька")
    parser.add_argument(
        "--use-cnn",
        action="store_true",
        help="Дополнительно запустить ResNet (финальный класс — по правилам ТЗ)",
    )
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--talc-threshold",
        default="auto",
        help="Порог U-Net: auto или число, например 0.12",
    )
    parser.add_argument(
        "--talc-heuristic",
        choices=["legacy"],
        default="legacy",
        help="Эвристика талька: старая тёмно-серая фаза",
    )
    parser.add_argument(
        "--unet-max-side",
        type=int,
        default=1536,
        help="Максимальная сторона для U-Net-инференса; 0 = полный размер без уменьшения",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Файл не найден: {args.image}")
        return 1

    talc_threshold = parse_talc_threshold(args.talc_threshold)

    out = analyze_image(
        args.image,
        talc_model=args.talc_model,
        classifier_model=args.classifier_model,
        talc_threshold=talc_threshold,
        talc_heuristic=args.talc_heuristic,
        unet_max_side=args.unet_max_side or None,
        use_unet=not args.no_unet,
        use_classifier=args.use_cnn and not args.no_classifier,
    )
    result = out["result"]
    m = result.metrics

    print(result.summary)
    print()
    print(format_metrics_table(m))
    print(f"\n  Метод классификации: {result.classification_method} (экспертные правила ТЗ)")
    if out["talc_threshold_used"] is not None:
        print(f"  Порог U-Net: {out['talc_threshold_used']:.3f}")

    if result.classifier_probabilities:
        classifier = out.get("classifier") or {}
        cnn_class = classifier.get("pred_label_ru") or classifier.get("pred_label") or "—"
        print(format_classifier_note(cnn_class, float(result.classifier_confidence or 0)))
        print(f"  Вероятности CNN: {result.classifier_probabilities}")

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
            "classification_method": result.classification_method,
            "rules_class": result.rules_class,
            "rules_class_ru": result.rules_class_ru,
            "classifier_confidence": result.classifier_confidence,
            "classifier_probabilities": result.classifier_probabilities,
            "talc_model": str(args.talc_model) if not args.no_unet else None,
            "talc_threshold_used": out["talc_threshold_used"],
            "classifier_model": str(args.classifier_model) if args.use_cnn else None,
            "metrics": m.__dict__,
            "metrics_table": format_metrics_table(m),
        }
        with open(args.output / f"{stem}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\nСохранено в {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

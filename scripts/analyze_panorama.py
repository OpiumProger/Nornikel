"""
Анализ панорамных OM-снимков.

Примеры:
    python scripts/analyze_panorama.py --list
    python scripts/analyze_panorama.py data/raw/Панорамы/11.jpg
    python scripts/analyze_panorama.py --all --limit 3 -o reports/panoramas
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io import imwrite
from src.metrics import format_metrics_table
from src.panorama import (
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_PANORAMA_DIR,
    DEFAULT_TALC_MODEL,
    analyze_panorama,
    list_panoramas,
)


def parse_talc_threshold(value: str) -> float | str:
    if value.lower() == "auto":
        return "auto"
    return float(value)


def print_summary(out: dict) -> None:
    result = out["result"]
    m = result.metrics
    pan = out["panorama"]

    print(f"Файл: {Path(out['input']).name}")
    print(
        f"  Размер: {pan['original_width']}x{pan['original_height']} "
        f"({pan['megapixels_original']} MP) -> рабочий {pan['working_width']}x{pan['working_height']} "
        f"({pan['megapixels_working']} MP, scale={pan['scale']})"
    )
    print(result.summary)
    print()
    print(format_metrics_table(m))
    print(f"\n  Метод классификации: {result.classification_method} (экспертные правила ТЗ)")
    if out.get("panorama", {}).get("tiles_total"):
        print(f"  U-Net тайлов (склейка): {out['panorama']['tiles_total']}")
    if out["talc_threshold_used"] is not None:
        print(f"  Порог U-Net: {out['talc_threshold_used']:.3f}")
    if result.classifier_probabilities:
        print(f"  CNN (для сравнения): {result.classifier_probabilities}")


def save_outputs(out: dict, output_dir: Path, preview_side: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(out["input"]).stem
    result = out["result"]
    m = result.metrics

    imwrite(output_dir / f"{stem}_overlay_preview.jpg", out["overlay_preview"])
    imwrite(output_dir / f"{stem}_talc.png", out["masks"]["talc"])
    imwrite(output_dir / f"{stem}_ordinary.png", out["masks"]["ordinary"])
    imwrite(output_dir / f"{stem}_fine.png", out["masks"]["fine"])
    if out["prob_map"] is not None:
        prob = (out["prob_map"] * 255).clip(0, 255).astype("uint8")
        imwrite(output_dir / f"{stem}_talc_probability.png", prob)

    report = {
        "file": Path(out["input"]).name,
        "panorama": out["panorama"],
        "ore_class": result.ore_class,
        "ore_class_ru": result.ore_class_ru,
        "summary": result.summary,
        "classification_method": result.classification_method,
        "rules_class": result.rules_class,
        "rules_class_ru": result.rules_class_ru,
        "classifier_confidence": result.classifier_confidence,
        "classifier_probabilities": result.classifier_probabilities,
        "classifier_tiles_used": (out.get("classifier") or {}).get("tiles_used"),
        "talc_threshold_used": out["talc_threshold_used"],
        "metrics": m.__dict__,
        "metrics_table": format_metrics_table(m),
    }
    with open(output_dir / f"{stem}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Анализ панорамных OM-снимков")
    parser.add_argument("image", type=Path, nargs="?", help="Путь к панораме")
    parser.add_argument("--panorama-dir", type=Path, default=DEFAULT_PANORAMA_DIR)
    parser.add_argument("--output", "-o", type=Path, default=Path("reports/panoramas"))
    parser.add_argument("--all", action="store_true", help="Обработать все панорамы в папке")
    parser.add_argument("--list", action="store_true", help="Показать список панорам")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=4096, help="Разрешение для сульфидов/CNN/оверлея")
    parser.add_argument("--preview-side", type=int, default=2400, help="Длинная сторона превью-оверлея")
    parser.add_argument(
        "--talc-mode",
        choices=("tiled_stitch", "downscale"),
        default="tiled_stitch",
        help="tiled_stitch: нарезка панорамы + склейка U-Net; downscale: быстрый режим",
    )
    parser.add_argument("--talc-tile-size", type=int, default=1536, help="Размер тайла U-Net на исходной панораме")
    parser.add_argument("--talc-tile-stride", type=int, default=1152, help="Шаг нарезки тайлов U-Net")
    parser.add_argument("--stitch-max-side", type=int, default=8192, help="Разрешение склеенной маски талька")
    parser.add_argument("--talc-model", type=Path, default=DEFAULT_TALC_MODEL)
    parser.add_argument("--classifier-model", type=Path, default=DEFAULT_CLASSIFIER_MODEL)
    parser.add_argument("--no-unet", action="store_true")
    parser.add_argument("--use-cnn", action="store_true", help="Дополнительно ResNet для сравнения")
    parser.add_argument("--no-classifier", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--talc-threshold", default="auto")
    parser.add_argument("--classifier-tile-size", type=int, default=768)
    parser.add_argument("--classifier-stride", type=int, default=512)
    parser.add_argument("--csv", type=Path, default=None, help="Сводный CSV при пакетном запуске")
    args = parser.parse_args()

    panoramas = list_panoramas(args.panorama_dir)
    if args.list:
        print(f"Панорамы в {args.panorama_dir}: {len(panoramas)}")
        for path in panoramas:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.name} ({size_mb:.1f} MB)")
        return 0

    if args.all:
        targets = panoramas
        if args.limit > 0:
            targets = targets[: args.limit]
    elif args.image:
        targets = [args.image]
    else:
        parser.error("Укажите файл, --all или --list")
        return 2

    if not targets:
        print("Панорамы не найдены.")
        return 1

    talc_threshold = parse_talc_threshold(args.talc_threshold)
    csv_rows = []
    t0 = time.time()

    for i, path in enumerate(targets, 1):
        if not path.exists():
            print(f"[skip] Не найден: {path}")
            continue

        print("=" * 60)
        print(f"[{i}/{len(targets)}] Анализ {path.name} ...", flush=True)
        item_t0 = time.time()

        out = analyze_panorama(
            path,
            max_side=args.max_side,
            preview_side=args.preview_side,
            talc_model=args.talc_model,
            classifier_model=args.classifier_model,
            talc_threshold=talc_threshold,
            use_unet=not args.no_unet,
            use_classifier=args.use_cnn and not args.no_classifier,
            classifier_tile_size=args.classifier_tile_size,
            classifier_stride=args.classifier_stride,
            talc_mode=args.talc_mode,
            talc_tile_size=args.talc_tile_size,
            talc_tile_stride=args.talc_tile_stride,
            stitch_max_side=args.stitch_max_side,
        )
        save_outputs(out, args.output, args.preview_side)
        print_summary(out)
        print(f"  Время: {(time.time() - item_t0):.1f} с")
        print(f"  Сохранено: {args.output}")

        result = out["result"]
        m = result.metrics
        csv_rows.append(
            {
                "file": path.name,
                "original_mp": out["panorama"]["megapixels_original"],
                "working_mp": out["panorama"]["megapixels_working"],
                "ore_class": result.ore_class,
                "ore_class_ru": result.ore_class_ru,
                "talc_pct": m.talc_pct,
                "sulfide_pct": m.sulfide_pct,
                "ordinary_pct": m.ordinary_pct,
                "fine_pct": m.fine_pct,
                "ordinary_share": m.ordinary_share,
                "fine_share": m.fine_share,
                "talc_mode": out["panorama"].get("talc_mode"),
                "talc_tiles": out["panorama"].get("tiles_total"),
                "cnn_confidence": result.classifier_confidence,
                "cnn_otalkovannaya": (result.classifier_probabilities or {}).get("otalkovannaya"),
                "cnn_ryadovaya": (result.classifier_probabilities or {}).get("ryadovaya"),
                "cnn_trudnoobogatimaya": (result.classifier_probabilities or {}).get("trudnoobogatimaya"),
                "cnn_tiles": (out.get("classifier") or {}).get("tiles_used"),
            }
        )

    if args.csv and csv_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nСводка CSV: {args.csv}")

    print("=" * 60)
    print(f"Готово: {len(csv_rows)} панорам за {(time.time() - t0) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

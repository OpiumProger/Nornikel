"""
Пакетный анализ + сравнение с метками папок.
Извлечение масок талька из синей разметки для обучения.

Использование:
    python scripts/batch_analyze.py
    python scripts/extract_talc_masks.py
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io import imread, imwrite
from src.metrics import classify_ore, compute_metrics
from src.preprocess import preprocess
from src.sulfide_mask import extract_sulfide_masks
from src.talc_mask import extract_talc_mask
from src.visualize import create_overlay

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def run_batch(manifest: Path, out_dir: Path, limit: int | None = None) -> None:
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    if limit:
        rows = rows[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, row in enumerate(rows):
        path = Path(row["processed_path"])
        if not path.exists():
            path = Path(row["original_path"])
        if not path.exists():
            continue

        img = preprocess(imread(path))
        talc_mask, talc_method = extract_talc_mask(img)
        ordinary, fine, _ = extract_sulfide_masks(img)
        metrics = compute_metrics(talc_mask, ordinary, fine, talc_method)
        result = classify_ore(metrics)

        true_label = row["label"]
        pred = result.ore_class
        match = true_label == pred

        results.append(
            {
                "filename": row["filename"],
                "true_label": true_label,
                "pred_label": pred,
                "match": match,
                "talc_pct": metrics.talc_pct,
                "talc_method": talc_method,
                "ordinary_share": metrics.ordinary_share,
                "fine_share": metrics.fine_share,
                "summary": result.summary,
            }
        )

        if i < 12:
            overlay = create_overlay(img, ordinary, fine, talc_mask)
            imwrite(out_dir / "samples" / f"{row['filename']}_overlay.jpg", overlay)

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(rows)}")

    csv_out = out_dir / "batch_results.csv"
    with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    correct = sum(1 for r in results if r["match"])
    by_class: dict[str, dict] = {}
    for r in results:
        c = r["true_label"]
        by_class.setdefault(c, {"n": 0, "ok": 0})
        by_class[c]["n"] += 1
        if r["match"]:
            by_class[c]["ok"] += 1

    ann = sum(1 for r in results if r["talc_method"] == "blue_annotation")

    print("=" * 60)
    print(f"Проанализировано: {total}")
    print(f"Точность vs метка папки: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"С синей разметкой талька: {ann}")
    for cls, st in sorted(by_class.items()):
        print(f"  {cls}: {st['ok']}/{st['n']}")
    print(f"Отчёт: {csv_out}")
    print(f"Примеры: {out_dir / 'samples'}")
    print("=" * 60)


def extract_training_masks(manifest: Path, out_dir: Path) -> None:
    """Сохраняет маски талька там, где есть синяя разметка."""
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    out_masks = out_dir / "talc_masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    saved = 0
    for row in rows:
        path = Path(row["processed_path"])
        if not path.exists():
            path = Path(row["original_path"])
        img = imread(path)
        talc_mask, method = extract_talc_mask(img)
        if method != "blue_annotation":
            continue
        imwrite(out_masks / f"{row['filename']}_talc.png", talc_mask)
        saved += 1

    print(f"Извлечено масок талька (синяя разметка): {saved}")
    print(f"Папка: {out_masks}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "extract_masks"], default="batch")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent
    manifest = project / "data" / "processed" / "manifest.csv"
    out_dir = project / "reports" / "pipeline"

    if args.mode == "batch":
        run_batch(manifest, out_dir, limit=args.limit)
    else:
        extract_training_masks(manifest, project / "data" / "processed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

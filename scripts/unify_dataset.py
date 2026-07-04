"""
Унификация датасета: ч1 + ч2 -> data/processed с едиными метками.

Структура:
    data/processed/
        images/
            otalkovannaya/
            ryadovaya/
            trudnoobogatimaya/
        manifest.csv

Использование:
    python unify_dataset.py
    python unify_dataset.py --copy          # копировать файлы (по умолчанию — hardlink)
    python unify_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image

# Импорт общей логики
from dataset_utils import LABEL_NAMES_RU, iter_images

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def unique_name(stem: str, suffix: str, used: set[str]) -> str:
    name = f"{stem}{suffix}"
    if name not in used:
        used.add(name)
        return name
    i = 2
    while True:
        candidate = f"{stem}__{i}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def place_file(src: Path, dst: Path, use_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os_link = __import__("os")
            os_link.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Унификация датасета руд")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--copy", action="store_true", help="Копировать вместо hardlink")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else project_dir / "data" / "raw"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else project_dir / "data" / "processed"
    images_dir = out_dir / "images"

    records = list(iter_images(raw_dir))
    if not records:
        print(f"Изображения не найдены в {raw_dir}")
        return 1

    print("=" * 60)
    print("УНИФИКАЦИЯ ДАТАСЕТА")
    print(f"Источник: {raw_dir}")
    print(f"Выход:    {out_dir}")
    print(f"Режим:    {'dry-run' if args.dry_run else ('copy' if args.copy else 'hardlink/copy-fallback')}")
    print(f"Записей:  {len(records)}")
    print("=" * 60)

    # Имена должны быть уникальны глобально: маски талька лежат в одной папке.
    used_names: set[str] = set()
    manifest_rows = []
    label_counter = Counter()

    for rec in records:
        label = rec["label"]
        if label == "unknown":
            print(f"  [skip] неизвестная метка: {rec['path']}")
            continue

        src = rec["path"]
        part_slug = "ch1" if "ч1" in rec["source_part"] else "ch2" if "ч2" in rec["source_part"] else "other"
        stem = f"{part_slug}__{src.stem}"
        suffix = src.suffix.lower()
        filename = unique_name(stem, suffix, used_names)
        dst = images_dir / label / filename

        width = height = None
        if not args.dry_run:
            place_file(src, dst, use_copy=args.copy)
            try:
                with Image.open(dst if dst.exists() else src) as im:
                    width, height = im.size
            except Exception:
                pass
        else:
            try:
                with Image.open(src) as im:
                    width, height = im.size
            except Exception:
                pass

        manifest_rows.append(
            {
                "filename": filename,
                "label": label,
                "label_ru": LABEL_NAMES_RU[label],
                "source_part": rec["source_part"],
                "source_folder": rec["source_folder"],
                "original_path": str(src),
                "processed_path": str(dst),
                "width": width,
                "height": height,
            }
        )
        label_counter[label] += 1

    if not args.dry_run:
        manifest_path = out_dir / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        # splits stub for train/val
        splits_path = out_dir / "splits.csv"
        with open(splits_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label", "split"])
            writer.writeheader()
            # простое 80/20 по каждому классу (детерминированно по имени)
            by_label: dict[str, list] = {}
            for row in manifest_rows:
                by_label.setdefault(row["label"], []).append(row)
            for label, rows in by_label.items():
                rows_sorted = sorted(rows, key=lambda r: r["filename"])
                n_val = max(1, len(rows_sorted) // 5)
                for i, row in enumerate(rows_sorted):
                    split = "val" if i < n_val else "train"
                    writer.writerow({"filename": row["filename"], "label": label, "split": split})

    print("\nИтог по классам:")
    for label, n in sorted(label_counter.items()):
        print(f"  {LABEL_NAMES_RU[label]:20} ({label}): {n}")

    if not args.dry_run:
        print(f"\nmanifest: {out_dir / 'manifest.csv'}")
        print(f"splits:   {out_dir / 'splits.csv'}")
        print(f"images:   {images_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

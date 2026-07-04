"""
Разведка датасета: статистика, примеры, поиск цветной разметки (тальк и др.).

Использование:
    python explore_dataset.py
    python explore_dataset.py --raw-dir "../data/raw"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dataset_utils import FOLDER_TO_LABEL, IMAGE_EXTENSIONS, LABEL_NAMES_RU, iter_images
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


def detect_colored_markup(img_bgr: np.ndarray) -> dict[str, float]:
    """Ищет яркие насыщенные линии разметки (тальк и др.)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total = img_bgr.shape[0] * img_bgr.shape[1]

    ranges = {
        "blue_cyan": [(90, 80, 80), (130, 255, 255)],
        "green": [(35, 80, 80), (85, 255, 255)],
        "red": [(0, 80, 80), (10, 255, 255), (170, 80, 80), (180, 255, 255)],
        "magenta": [(140, 80, 80), (170, 255, 255)],
        "yellow": [(20, 80, 80), (35, 255, 255)],
    }

    result = {}
    for name, bounds in ranges.items():
        mask = None
        for i in range(0, len(bounds), 2):
            lo, hi = np.array(bounds[i]), np.array(bounds[i + 1])
            part = cv2.inRange(hsv, lo, hi)
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        pct = float(np.count_nonzero(mask)) / total * 100
        result[name] = round(pct, 4)

    result["any_markup"] = round(max(result.values()), 4)
    result["likely_annotated"] = result["any_markup"] > 0.05
    return result


def image_stats(path: Path) -> dict:
    row = {
        "file": path.name,
        "path": str(path),
        "width": None,
        "height": None,
        "megapixels": None,
        "file_mb": round(path.stat().st_size / 1024**2, 3),
        "error": None,
    }
    try:
        with Image.open(path) as im:
            w, h = im.size
            row["width"] = w
            row["height"] = h
            row["megapixels"] = round(w * h / 1e6, 2)
            row["mode"] = im.mode
    except Exception as exc:
        row["error"] = str(exc)
    return row


def make_montage(items: list[tuple[Path, str]], out_path: Path, thumb: int = 320) -> None:
    """Сетка превью: path + подпись."""
    if not items:
        return

    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * (thumb + 28)), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for idx, (path, caption) in enumerate(items):
        r, c = divmod(idx, cols)
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                x = c * thumb + (thumb - im.width) // 2
                y = r * (thumb + 28) + (thumb - im.height) // 2
                canvas.paste(im, (x, y))
        except Exception:
            draw.text((c * thumb + 10, r * (thumb + 28) + 10), "ERR", fill=(255, 80, 80))

        draw.text((c * thumb + 6, r * (thumb + 28) + thumb + 4), caption[:40], fill=(220, 220, 220))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка датасета руд")
    parser.add_argument("--raw-dir", default=None, help="Папка data/raw")
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--markup-check", type=int, default=80, help="Сколько фото проверить на разметку")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else project_dir / "data" / "raw"
    reports_dir = project_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_images(raw_dir))
    if not records:
        print(f"Изображения не найдены в {raw_dir}")
        return 1

    print("=" * 60)
    print("РАЗВЕДКА ДАТАСЕТА")
    print(f"Папка: {raw_dir}")
    print(f"Всего изображений: {len(records)}")
    print("=" * 60)

    # --- Статистика по классам ---
    by_label = Counter(r["label"] for r in records)
    by_part = Counter(r["source_part"] for r in records)
    by_folder = Counter((r["source_part"], r["source_folder"]) for r in records)

    print("\nПо классам:")
    for label, n in sorted(by_label.items()):
        ru = LABEL_NAMES_RU.get(label, label)
        print(f"  {ru:20} ({label}): {n}")

    print("\nПо частям датасета:")
    for part, n in sorted(by_part.items()):
        print(f"  {part}: {n}")

    print("\nИсходные папки:")
    for (part, folder), n in sorted(by_folder.items()):
        print(f"  {part} / {folder}: {n}")

    # --- Детальная таблица ---
    detailed = []
    res_counter = Counter()
    errors = 0

    for rec in records:
        st = image_stats(rec["path"])
        st.update(
            {
                "label": rec["label"],
                "source_part": rec["source_part"],
                "source_folder": rec["source_folder"],
            }
        )
        if st["error"]:
            errors += 1
        else:
            res_counter[f"{st['width']}x{st['height']}"] += 1
        detailed.append(st)

    csv_path = reports_dir / "dataset_inventory.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "label",
                "source_part",
                "source_folder",
                "width",
                "height",
                "megapixels",
                "file_mb",
                "mode",
                "path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed)

    print(f"\nРазрешения (топ):")
    for res, n in res_counter.most_common(5):
        print(f"  {res}: {n}")

    if errors:
        print(f"\nОшибки чтения: {errors}")

    # --- Проверка цветной разметки ---
    print(f"\nПроверка цветной разметки (до {args.markup_check} фото)...")
    markup_rows = []
    annotated_count = 0

    # Берём пропорционально из каждого класса
    per_label = defaultdict(list)
    for rec in records:
        per_label[rec["label"]].append(rec)

    check_list = []
    for label, items in per_label.items():
        step = max(1, len(items) // (args.markup_check // max(len(per_label), 1) + 1))
        check_list.extend(items[::step])

    check_list = check_list[: args.markup_check]

    for rec in check_list:
        img = read_image_bgr(rec["path"])
        if img is None:
            continue
        mk = detect_colored_markup(img)
        row = {
            "file": rec["path"].name,
            "label": rec["label"],
            "source_part": rec["source_part"],
            **mk,
        }
        markup_rows.append(row)
        if mk["likely_annotated"]:
            annotated_count += 1

    markup_csv = reports_dir / "markup_scan.csv"
    if markup_rows:
        with open(markup_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(markup_rows[0].keys()))
            writer.writeheader()
            writer.writerows(markup_rows)

    print(f"  Проверено: {len(markup_rows)}")
    print(f"  С вероятной разметкой (>0.05% ярких линий): {annotated_count}")

    if markup_rows:
        color_totals = Counter()
        for row in markup_rows:
            for color in ("blue_cyan", "green", "red", "magenta", "yellow"):
                if row[color] > 0.05:
                    color_totals[color] += 1
        if color_totals:
            print("  Доминирующие цвета разметки:")
            for color, n in color_totals.most_common():
                print(f"    {color}: {n} фото")

    # --- Превью-монтажи ---
    samples_dir = reports_dir / "samples"
    for label in sorted(by_label.keys()):
        items = [r["path"] for r in records if r["label"] == label]
        picked = []
        step = max(1, len(items) // args.samples_per_class)
        for p in items[::step][: args.samples_per_class]:
            picked.append((p, p.name))
        make_montage(
            picked,
            samples_dir / f"{label}_samples.jpg",
        )

    # --- JSON-отчёт ---
    summary = {
        "total_images": len(records),
        "by_label": dict(by_label),
        "by_part": dict(by_part),
        "by_folder": {f"{a}|{b}": n for (a, b), n in by_folder.items()},
        "resolutions": dict(res_counter.most_common()),
        "read_errors": errors,
        "markup_checked": len(markup_rows),
        "markup_likely_annotated": annotated_count,
        "label_names_ru": LABEL_NAMES_RU,
    }
    json_path = reports_dir / "dataset_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- Markdown-отчёт ---
    md_path = reports_dir / "dataset_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Отчёт по датасету\n\n")
        f.write(f"**Всего изображений:** {len(records)}\n\n")
        f.write("## Распределение по классам\n\n")
        f.write("| Класс | Метка | Кол-во |\n|-------|-------|--------|\n")
        for label, n in sorted(by_label.items()):
            f.write(f"| {LABEL_NAMES_RU.get(label, label)} | `{label}` | {n} |\n")

        f.write("\n## Части датасета\n\n")
        for part, n in sorted(by_part.items()):
            f.write(f"- **{part}**: {n} фото\n")

        f.write("\n## Разрешения\n\n")
        for res, n in res_counter.most_common():
            f.write(f"- {res}: {n}\n")

        f.write("\n## Цветная разметка\n\n")
        f.write(
            f"Проверено {len(markup_rows)} фото. "
            f"Вероятная разметка найдена на **{annotated_count}** снимках.\n\n"
        )
        f.write("Примеры по классам: `reports/samples/`\n")

    print("\n" + "=" * 60)
    print("Готово. Отчёты:")
    print(f"  {csv_path}")
    print(f"  {markup_csv if markup_rows else '(разметка — нет данных)'}")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print(f"  {samples_dir}/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

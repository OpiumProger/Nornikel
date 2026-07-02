"""Общие утилиты для работы с датасетом."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

FOLDER_TO_LABEL = {
    "оталькованные": "otalkovannaya",
    "оталькованные руды": "otalkovannaya",
    "рядовые": "ryadovaya",
    "рядовые руды": "ryadovaya",
    "тонкие": "trudnoobogatimaya",
    "труднообогатимые руды": "trudnoobogatimaya",
}

LABEL_NAMES_RU = {
    "otalkovannaya": "оталькованная",
    "ryadovaya": "рядовая",
    "trudnoobogatimaya": "труднообогатимая",
}


def iter_images(raw_dir: Path):
    """Обходит ч1/ч2 и возвращает записи об изображениях."""
    for part_dir in sorted(raw_dir.iterdir()):
        if not part_dir.is_dir() or part_dir.name == "Панорамы":
            continue

        for img_path in sorted(part_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rel = img_path.relative_to(part_dir)
            label_folder = None
            for parent in rel.parents:
                if parent == Path("."):
                    break
                name_key = parent.name.lower()
                if name_key in FOLDER_TO_LABEL:
                    label_folder = parent.name
                    break
            if label_folder is None and len(rel.parts) > 1:
                label_folder = rel.parts[0]
            elif label_folder is None:
                continue

            label_key = FOLDER_TO_LABEL.get(label_folder.lower(), "unknown")

            yield {
                "path": img_path,
                "source_part": part_dir.name,
                "source_folder": label_folder,
                "label": label_key,
            }

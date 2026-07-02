"""
Скачивание файлов с публичной ссылки Яндекс.Диска (рекурсивно по папкам).

Использование:
    python download_yadisk.py "https://disk.yandex.ru/d/XXXXX"
    python download_yadisk.py "https://disk.yandex.ru/d/XXXXX" --output "../data/raw"
"""

from __future__ import annotations

import argparse
import io
import sys
import time

# Windows-консоль: кириллица и спецсимволы в print
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

API_RESOURCES = "https://cloud-api.yandex.net/v1/disk/public/resources"
API_DOWNLOAD = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

# Расширения изображений из ТЗ
IMAGE_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_yadisk_url(url: str) -> tuple[str, str]:
    """
    Разбирает публичную ссылку Яндекс.Диска.
    Возвращает (public_key, remote_path).
    Пример:
      https://disk.yandex.ru/d/HASH/подпапка
      -> public_key=https://disk.yandex.ru/d/HASH, remote_path=/подпапка
    """
    url = url.strip()
    if not url.startswith("http"):
        url = f"https://disk.yandex.ru/d/{url}"

    parsed = urlparse(url)
    path = unquote(parsed.path).strip("/")
    parts = path.split("/", 1)

    if len(parts) < 2 or parts[0] != "d":
        raise ValueError(f"Неверный формат ссылки: {url}")

    hash_and_rest = parts[1].split("/", 1)
    disk_hash = hash_and_rest[0]
    public_key = f"{parsed.scheme}://{parsed.netloc}/d/{disk_hash}"
    remote_path = f"/{hash_and_rest[1]}" if len(hash_and_rest) > 1 else "/"
    return public_key, remote_path


def api_get(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [!] Ошибка API, повтор через {wait}с: {exc}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def list_folder(public_key: str, remote_path: str = "/") -> list[dict]:
    items: list[dict] = []
    offset = 0
    limit = 200

    while True:
        data = api_get(
            API_RESOURCES,
            {
                "public_key": public_key,
                "path": remote_path,
                "limit": limit,
                "offset": offset,
                "fields": "_embedded.items.name,_embedded.items.path,_embedded.items.type,_embedded.items.mime_type,_embedded.items.size",
            },
        )
        embedded = data.get("_embedded", {})
        batch = embedded.get("items", [])
        items.extend(batch)

        total = embedded.get("total", len(batch))
        offset += len(batch)
        if offset >= total or not batch:
            break

    return items


def download_file(public_key: str, remote_path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] уже есть: {dest.name}")
        return

    data = api_get(API_DOWNLOAD, {"public_key": public_key, "path": remote_path})
    href = data["href"]

    for attempt in range(5):
        try:
            with requests.get(href, stream=True, timeout=300) as response:
                response.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            print(f"  [ok]   {dest.relative_to(dest.parents[len(dest.parents) - 1])}")
            return
        except requests.RequestException as exc:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"  [!] Ошибка загрузки {dest.name}, повтор через {wait}с: {exc}")
            time.sleep(wait)


def is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def walk_and_download(
    public_key: str,
    remote_path: str,
    local_dir: Path,
    images_only: bool = True,
) -> tuple[int, int]:
    downloaded = 0
    skipped = 0

    items = list_folder(public_key, remote_path)
    for item in items:
        name = item["name"]
        item_path = item["path"]
        local_path = local_dir / name

        if item["type"] == "dir":
            print(f"\n[папка] {name}")
            sub_dl, sub_sk = walk_and_download(
                public_key, item_path, local_path, images_only=images_only
            )
            downloaded += sub_dl
            skipped += sub_sk
        else:
            if images_only and not is_image(name):
                print(f"  [skip] не изображение: {name}")
                skipped += 1
                continue

            size_mb = item.get("size", 0) / (1024 * 1024)
            print(f"  [dl]   {name} ({size_mb:.1f} MB)")
            before = local_path.exists()
            download_file(public_key, item_path, local_path)
            if not before and local_path.exists():
                downloaded += 1
            else:
                skipped += 1

    return downloaded, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Скачать данные с публичного Яндекс.Диска")
    parser.add_argument("url", help="Публичная ссылка на папку или файл")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Локальная папка для сохранения (по умолчанию: ../data/raw)",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Скачивать все файлы, не только изображения",
    )
    args = parser.parse_args()

    public_key, remote_path = parse_yadisk_url(args.url)

    script_dir = Path(__file__).resolve().parent
    default_output = script_dir.parent / "data" / "raw"
    output_dir = Path(args.output).resolve() if args.output else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Яндекс.Диск -> локальная папка")
    print(f"Ссылка:  {args.url.strip()}")
    print(f"Ключ:    {public_key}")
    print(f"Путь:    {remote_path}")
    print(f"Куда:    {output_dir}")
    print(f"Фильтр:  {'все файлы' if args.all_files else 'только изображения'}")
    print("=" * 60)

    # Проверяем доступ к ресурсу
    root = api_get(API_RESOURCES, {"public_key": public_key, "path": remote_path})
    root_name = root.get("name", "disk")
    print(f"\nЦелевая папка: {root_name}")

    # Если ссылка на один файл — скачиваем напрямую
    if root.get("type") == "file":
        dest = output_dir / root_name
        download_file(public_key, remote_path, dest)
        print(f"\nГотово: 1 файл -> {dest}")
        return 0

    # Ссылка на папку
    target = output_dir / root_name
    downloaded, skipped = walk_and_download(
        public_key,
        remote_path,
        target,
        images_only=not args.all_files,
    )

    print("\n" + "=" * 60)
    print(f"Загружено новых файлов: {downloaded}")
    print(f"Пропущено:              {skipped}")
    print(f"Папка:                  {target}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as exc:
        print(f"\nОшибка HTTP: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text, file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        raise SystemExit(130)

# Норникель Hackathon: классификация руд по OM-снимкам

Legacy README для команды: что уже сделано, как воспроизвести пайплайн и что сейчас находится в проекте.

## Цель

End-to-end система для анализа микрофотографий руд:

- сегментация талька;
- эвристическое выделение сульфидов и разделение на обычные/тонкие срастания;
- расчёт долей фаз;
- экспертная классификация руды;
- обучение CNN-классификатора на 3 класса руды.

Классы:

- `otalkovannaya` — оталькованная руда;
- `ryadovaya` — рядовая руда;
- `trudnoobogatimaya` — труднообогатимая руда.

## Текущий статус

Данные скачаны с Яндекс.Диска и унифицированы локально.

Сейчас в `data/processed`:

- `manifest.csv` — полный список изображений, меток и путей;
- `splits.csv` — базовый train/val split;
- `images/` — изображения по 3 классам;
- `talc_masks/` — автоматически извлечённые маски талька из синей разметки (для обучения U-Net).

Статистика:

- всего изображений: `707`;
- масок талька для обучения U-Net: `332`;
- U-Net талька: val Dice ≈ **0.55** (порог 0.05, `models/talc_unet_config.json`);
- ResNet18-классификатор: val acc ≈ **88%**, macro F1 ≈ **0.88**;
- панорамы: 14 шт. в `data/raw/Панорамы/`, прогон batch-анализа готов.

**Тальк в продакшене:** гибрид `U-Net OR dark_gray_phase` (синяя разметка на инференсе не используется).  
Подробно: [`docs/talc_detection.md`](docs/talc_detection.md).

**Финальный класс руды** — экспертные правила ТЗ (`talc_pct > 10%` → оталькованная). CNN — только для сравнения.

## Структура проекта

```text
.
├── analyze.py                  # end-to-end анализ одного снимка
├── classify_image.py           # инференс ResNet-классификатора
├── train.py                    # обучение U-Net для талька
├── train_classifier.py         # обучение ResNet18 на 3 класса руды
├── run_train.bat               # быстрый запуск обучения U-Net
├── requirements.txt
├── src/
│   ├── io.py                   # чтение/запись изображений с кириллицей в путях
│   ├── preprocess.py           # CLAHE + фильтрация
│   ├── talc_mask.py            # dark_gray_phase + legacy синяя разметка
│   ├── talc_unet.py            # инференс U-Net, тайлы, порог
│   ├── talc_pipeline.py        # гибрид U-Net + dark_gray_phase
│   ├── panorama_talc.py        # тайловая склейка талька на панорамах
│   ├── panorama.py             # оркестрация анализа панорам
│   ├── sulfide_mask.py         # эвристика сульфидов
│   ├── ore_classifier.py       # ResNet18 инференс (сравнение)
│   ├── metrics.py              # метрики и экспертные правила
│   └── visualize.py            # цветной overlay
├── scripts/
│   ├── download_yadisk.py      # скачивание публичных папок Яндекс.Диска
│   ├── explore_dataset.py      # разведка датасета
│   ├── unify_dataset.py        # унификация ч.1 + ч.2
│   ├── batch_analyze.py        # batch-анализ и извлечение talc masks
│   ├── analyze_panorama.py     # анализ панорам (downscale / tiled_stitch)
│   ├── calibrate_talc_unet.py  # калибровка порога U-Net
│   ├── dataset_utils.py
│   └── check_env.py
├── docs/
│   └── talc_detection.md       # описание определения доли талька
├── data/                       # НЕ коммитить в GitHub
├── models/                     # НЕ коммитить крупные .pth, лучше через cloud/release
└── reports/                    # отчёты/превью, обычно НЕ коммитить
```

## Окружение

Основное окружение:

```powershell
conda activate new_chemberta_env
```

Проверка:

```powershell
python scripts/check_env.py
```

На рабочей машине проверено:

- GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`, 4 GB VRAM;
- PyTorch: `2.5.1+cu121`;
- CUDA доступна;
- `segmentation_models_pytorch`, `albumentations`, `opencv-python-headless` установлены.

## Воспроизведение данных

Если данных нет локально:

```powershell
python scripts/download_yadisk.py "https://disk.yandex.ru/d/Fo5eIM984glHaA/Фото%20руд%20по%20сортам.%20ч1"
python scripts/download_yadisk.py "https://disk.yandex.ru/d/Fo5eIM984glHaA/Фото%20руд%20по%20сортам.%20ч2"
python scripts/download_yadisk.py "https://disk.yandex.ru/d/Fo5eIM984glHaA/Панорамы"
```

Потом:

```powershell
python scripts/explore_dataset.py
python scripts/unify_dataset.py
python scripts/batch_analyze.py --mode extract_masks
```

## Обучение U-Net талька

Запуск под RTX 3050:

```powershell
python train.py --epochs 25 --batch-size 1 --patch-size 256
```

Выход:

```text
models/best_talc_unet.pth
```

Последний успешный результат:

```text
best val_dice = 0.4924
```

Если есть больше VRAM:

```powershell
python train.py --epochs 25 --batch-size 2 --patch-size 384
```

## Обучение ResNet-классификатора

`torchvision` в текущем conda-env падает из-за PIL DLL, поэтому `train_classifier.py` содержит самописный ResNet18 на чистом PyTorch.

Запуск:

```powershell
python train_classifier.py --epochs 30 --batch-size 8 --image-size 224
```

Если CUDA OOM:

```powershell
python train_classifier.py --epochs 30 --batch-size 4 --image-size 224
```

Выход:

```text
models/best_ore_resnet18.pth
reports/classifier/train_log.csv
reports/classifier/confusion_matrix.csv
reports/classifier/per_class_metrics.json
```

Инференс одного изображения:

```powershell
python classify_image.py "path\to\image.jpg"
```

## End-to-End анализ одного снимка

```powershell
python analyze.py "path\to\image.jpg" -o reports/test
```

По умолчанию:

- гибрид талька: **U-Net + `dark_gray_phase`** (`src/talc_pipeline.py`);
- эвристика сульфидов;
- экспертные правила ТЗ (финальный класс):
  - если `talc_pct > 10%` → оталькованная;
  - иначе если обычные срастания преобладают → рядовая;
  - иначе → труднообогатимая.

## Анализ панорам

```powershell
python scripts/analyze_panorama.py --list
python scripts/analyze_panorama.py "data\raw\Панорамы\11.jpg" -o reports/panoramas --talc-mode downscale
python scripts/analyze_panorama.py --all --limit 5 -o reports/panoramas --talc-mode downscale --use-cnn
```

Режимы талька: `downscale` (быстро) / `tiled_stitch` (точнее, медленно).

Выход:

```text
*_overlay.jpg
*_talc.png
*_talc_probability.png
*_ordinary.png
*_fine.png
*_report.json
```

Fallback без U-Net:

```powershell
python analyze.py "path\to\image.jpg" -o reports/test --no-unet
```

## Цвета маски

- зелёный — обычные срастания;
- красный — тонкие срастания;
- синий — тальк.

## Что коммитить в GitHub

Коммитить (код + документация):

```text
README.md
docs/talc_detection.md
.gitignore
requirements.txt
analyze.py
classify_image.py
train.py
train_classifier.py
run_train.bat
src/                          # все модули, включая talc_pipeline, panorama*, ore_classifier
scripts/                      # включая analyze_panorama.py, calibrate_talc_unet.py
models/talc_unet_config.json  # порог и параметры U-Net (без весов .pth)
ore-classifier-hackathon-plan.md
```

Опционально (лёгкие отчёты):

```text
reports/dataset_report.md
reports/classifier/per_class_metrics.json
reports/classifier/confusion_matrix.csv
```

Не коммитить:

```text
data/raw/
data/processed/images/
data/processed/talc_masks/
models/*.pth
reports/**/*.jpg
reports/**/*.png
reports/**/summary.csv
*.log
```

Модели — через GitHub Release / облако:

```text
models/best_talc_unet.pth
models/best_ore_resnet18.pth
```

## Ближайший план

1. Streamlit UI (загрузка, overlay, метрики, экспорт).
2. Подкрутить эвристику сульфидов (крупные зёрна → обычные).
3. Демо на 3–5 минут к дедлайну 4 июля 23:59.

## Ограничения текущей версии

- U-Net обучена на псевдо-масках из синей разметки — слабая генерализация на панорамы.
- `dark_gray_phase` — эвристика, не ground truth.
- Сульфиды — эвристика; сейчас завышается доля «тонких» срастаний.
- CNN на панорамах ненадёжен; финальный класс — только правила ТЗ.

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
- `talc_masks/` — автоматически извлечённые маски талька из синей разметки.

Статистика:

- всего изображений: `707`;
- масок талька для обучения U-Net: `332`;
- U-Net талька обучена, лучший `val_dice = 0.4924`;
- ResNet-классификатор написан, smoke-test на 1 эпохе дал `val_macro_f1 ~= 0.55`.

Важно: метрики U-Net пока умеренные, потому что маски талька получены автоматически из контурной/цветной разметки, а не из идеальной пиксельной ground truth.

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
│   ├── talc_mask.py            # эвристика/разметка талька
│   ├── talc_unet.py            # инференс обученной U-Net
│   ├── sulfide_mask.py         # эвристика сульфидов
│   ├── metrics.py              # метрики и экспертные правила
│   └── visualize.py            # цветной overlay
├── scripts/
│   ├── download_yadisk.py      # скачивание публичных папок Яндекс.Диска
│   ├── explore_dataset.py      # разведка датасета
│   ├── unify_dataset.py        # унификация ч.1 + ч.2
│   ├── batch_analyze.py        # batch-анализ и извлечение talc masks
│   ├── dataset_utils.py
│   └── check_env.py
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

По умолчанию используется:

- U-Net талька из `models/best_talc_unet.pth`;
- эвристика сульфидов;
- экспертные правила:
  - если `talc_pct > 10%` → оталькованная;
  - иначе если обычные срастания преобладают → рядовая;
  - иначе → труднообогатимая.

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

Коммитить:

```text
README.md
.gitignore
requirements.txt
analyze.py
classify_image.py
train.py
train_classifier.py
run_train.bat
src/
scripts/
ore-classifier-hackathon-plan.md
```

Опционально можно коммитить лёгкие отчёты:

```text
reports/dataset_report.md
```

Не коммитить:

```text
data/raw/
data/processed/images/
data/processed/talc_masks/
models/*.pth
reports/**/*.jpg
reports/**/*.png
*.log
```

Модели лучше передавать отдельно через облако/GitHub Release:

```text
models/best_talc_unet.pth
models/best_ore_resnet18.pth
```

## Ближайший план

1. Дождаться обучения ResNet-классификатора.
2. Подключить `best_ore_resnet18.pth` в финальный `analyze.py` как дополнительный классификационный сигнал.
3. Сделать Streamlit UI:
   - загрузка изображения;
   - overlay маски;
   - таблица метрик;
   - текстовый вывод;
   - экспорт CSV/JSON/PDF.
4. Подготовить демо на 3-5 минут.

## Ограничения текущей версии

- U-Net обучается на псевдо-масках из цветной разметки, поэтому это не идеальная ground truth.
- Сульфиды пока выделяются эвристически, без обученной пиксельной модели.
- Финальная классификация должна быть усилена ResNet-классификатором на 3 класса.
- Панорамы скачаны, но пока не включены в обучение; их лучше использовать для демо тайлового инференса.

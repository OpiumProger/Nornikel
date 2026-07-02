r"""
Обучение ResNet18-классификатора руды на 3 класса.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST = PROJECT_DIR / "data" / "processed" / "manifest.csv"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports" / "classifier"

LABEL_TO_ID = {
    "otalkovannaya": 0,
    "ryadovaya": 1,
    "trudnoobogatimaya": 2,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

LABEL_RU = {
    "otalkovannaya": "оталькованная",
    "ryadovaya": "рядовая",
    "trudnoobogatimaya": "труднообогатимая",
}

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


def imread_rgb(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_manifest() -> list[dict]:
    rows = []
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            label = row["label"]
            if label not in LABEL_TO_ID:
                continue
            path = Path(row["processed_path"])
            if not path.exists():
                path = Path(row["original_path"])
            if not path.exists():
                continue
            rows.append(
                {
                    "path": path,
                    "label": label,
                    "label_id": LABEL_TO_ID[label],
                    "filename": row["filename"],
                    "source_part": row["source_part"],
                }
            )
    return rows


def stratified_split(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train_rows, val_rows = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def resize_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y = max(0, (h - size) // 2)
    x = max(0, (w - size) // 2)
    crop = img[y : y + size, x : x + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def random_resized_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    area = h * w

    for _ in range(10):
        target_area = random.uniform(0.55, 1.0) * area
        aspect = random.uniform(0.75, 1.33)
        crop_w = int(round((target_area * aspect) ** 0.5))
        crop_h = int(round((target_area / aspect) ** 0.5))

        if crop_w <= w and crop_h <= h:
            x = random.randint(0, w - crop_w)
            y = random.randint(0, h - crop_h)
            crop = img[y : y + crop_h, x : x + crop_w]
            return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    return center_crop(resize_short_side(img, size), size)


def color_jitter(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32) / 255.0

    brightness = random.uniform(0.85, 1.15)
    contrast = random.uniform(0.85, 1.15)
    out = out * brightness
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * contrast + mean

    if random.random() < 0.4:
        hsv = cv2.cvtColor((out.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.85, 1.15)
        out = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return (out.clip(0, 1) * 255).astype(np.uint8)


def normalize(img: np.ndarray) -> torch.Tensor:
    x = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1).float()


class OreDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int, train: bool):
        self.rows = rows
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = imread_rgb(row["path"])

        if self.train:
            img = random_resized_crop(img, self.image_size)
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1])
            if random.random() < 0.6:
                img = color_jitter(img)
            if random.random() < 0.15:
                img = cv2.GaussianBlur(img, (3, 3), 0)
        else:
            img = center_crop(resize_short_side(img, self.image_size), self.image_size)

        return normalize(img), torch.tensor(row["label_id"], dtype=torch.long)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    macro_f1: float
    confusion: np.ndarray
    per_class: dict[str, dict[str, float]]


def make_class_weights(rows: list[dict], device: torch.device) -> torch.Tensor:
    counts = Counter(row["label_id"] for row in rows)
    total = sum(counts.values())
    weights = []
    for i in range(len(LABEL_TO_ID)):
        weights.append(total / max(1, counts[i]))
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def f1_from_confusion(conf: np.ndarray) -> tuple[float, dict[str, dict[str, float]]]:
    per_class = {}
    f1s = []
    for i in range(conf.shape[0]):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        label = ID_TO_LABEL[i]
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(conf[i, :].sum()),
        }
        f1s.append(f1)
    return float(np.mean(f1s)), per_class


def train_one_epoch(model, loader, optimizer, criterion, device, scaler) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> EvalResult:
    model.eval()
    total_loss = 0.0
    conf = np.zeros((len(LABEL_TO_ID), len(LABEL_TO_ID)), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)

        for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            conf[int(true), int(pred)] += 1

    acc = float(np.trace(conf) / max(1, conf.sum()))
    macro_f1, per_class = f1_from_confusion(conf)
    return EvalResult(
        loss=total_loss / max(1, len(loader)),
        accuracy=acc,
        macro_f1=macro_f1,
        confusion=conf,
        per_class=per_class,
    )


def save_confusion(conf: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [ID_TO_LABEL[i] for i in range(len(LABEL_TO_ID))]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for i, label in enumerate(labels):
            writer.writerow([label, *conf[i].tolist()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ResNet18 ore classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    rows = load_manifest()
    train_rows, val_rows = stratified_split(rows, args.val_ratio, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("TRAIN ResNet18 (3 класса руды)")
    print(f"device: {device}", end="")
    if device.type == "cuda":
        print(f" — {torch.cuda.get_device_name(0)}")
    else:
        print()
    print(f"images: {len(rows)} | train: {len(train_rows)} | val: {len(val_rows)}")
    print("classes:")
    for label, idx in LABEL_TO_ID.items():
        print(f"  {idx}: {label} ({LABEL_RU[label]})")
    print("=" * 70)

    train_ds = OreDataset(train_rows, args.image_size, train=True)
    val_ds = OreDataset(val_rows, args.image_size, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ResNet18(num_classes=len(LABEL_TO_ID)).to(device)
    class_weights = make_class_weights(train_rows, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / "best_ore_resnet18.pth"
    log_path = REPORTS_DIR / "train_log.csv"

    best_f1 = 0.0
    start = time.time()

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val.loss,
                    "val_acc": val.accuracy,
                    "val_macro_f1": val.macro_f1,
                    "lr": lr,
                }
            )
            f.flush()

            print(
                f"epoch {epoch:02d}/{args.epochs} | "
                f"train_loss {train_loss:.4f} | val_loss {val.loss:.4f} | "
                f"val_acc {val.accuracy:.4f} | val_macro_f1 {val.macro_f1:.4f}",
                flush=True,
            )

            if val.macro_f1 > best_f1:
                best_f1 = val.macro_f1
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_macro_f1": val.macro_f1,
                        "val_accuracy": val.accuracy,
                        "image_size": args.image_size,
                        "label_to_id": LABEL_TO_ID,
                        "id_to_label": ID_TO_LABEL,
                    },
                    best_path,
                )
                save_confusion(val.confusion, REPORTS_DIR / "confusion_matrix.csv")
                with open(REPORTS_DIR / "per_class_metrics.json", "w", encoding="utf-8") as jf:
                    json.dump(val.per_class, jf, ensure_ascii=False, indent=2)
                print(f"  -> сохранено {best_path} (macro_f1={val.macro_f1:.4f})", flush=True)

    print("=" * 70)
    print(f"Готово за {(time.time() - start) / 60:.1f} мин. Лучший macro-F1: {best_f1:.4f}")
    print(f"Модель: {best_path}")
    print(f"Лог:    {log_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

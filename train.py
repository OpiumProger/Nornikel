"""
Обучение U-Net: сегментация талька.
Оптимизировано под RTX 3050 4GB.

Запуск (из папки Нонрникель):
    conda activate new_chemberta_env
    pip install segmentation-models-pytorch albumentations opencv-python-headless
    python train.py

    python train.py --epochs 30 --batch-size 2 --patch-size 384
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --- пути ---
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

IMAGES_DIR = PROJECT_DIR / "data" / "processed" / "images"
MASKS_DIR = PROJECT_DIR / "data" / "processed" / "talc_masks"
MANIFEST = PROJECT_DIR / "data" / "processed" / "manifest.csv"
MODELS_DIR = PROJECT_DIR / "models"

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


def load_pairs() -> list[dict]:
    """Сопоставляет изображения с масками талька."""
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Нет manifest: {MANIFEST}")

    pairs = []
    for row in csv.DictReader(open(MANIFEST, encoding="utf-8-sig")):
        stem = Path(row["filename"]).stem
        mask_path = MASKS_DIR / f"{row['filename']}_talc.png"
        if not mask_path.exists():
            continue
        img_path = Path(row["processed_path"])
        if not img_path.exists():
            img_path = Path(row["original_path"])
        if not img_path.exists():
            continue
        pairs.append(
            {
                "image": img_path,
                "mask": mask_path,
                "label": row["label"],
                "filename": row["filename"],
            }
        )
    return pairs


class TalcDataset(Dataset):
    def __init__(
        self,
        pairs: list[dict],
        patch_size: int,
        augment: bool = False,
        positive_crop_prob: float = 0.75,
    ):
        self.pairs = pairs
        self.patch_size = patch_size
        self.augment = augment
        self.positive_crop_prob = positive_crop_prob
        try:
            import albumentations as A

            self.transform = (
                A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02, p=0.5),
                        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
                    ]
                )
                if augment
                else None
            )
        except ImportError:
            self.transform = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item = self.pairs[idx]
        image = np.array(Image.open(item["image"]).convert("RGB"))
        mask = np.array(Image.open(item["mask"]).convert("L"))
        if mask.shape[:2] != image.shape[:2]:
            mask = np.array(
                Image.fromarray(mask).resize(
                    (image.shape[1], image.shape[0]),
                    Image.Resampling.NEAREST,
                )
            )
        mask = (mask > 127).astype(np.float32)

        h, w = image.shape[:2]
        ps = self.patch_size
        if h > ps and w > ps:
            if self.augment and random.random() < self.positive_crop_prob and mask.sum() > 0:
                ys, xs = np.where(mask > 0)
                center_idx = random.randrange(len(xs))
                cx, cy = int(xs[center_idx]), int(ys[center_idx])
                x = min(max(cx - ps // 2, 0), w - ps)
                y = min(max(cy - ps // 2, 0), h - ps)
            else:
                y = random.randint(0, h - ps)
                x = random.randint(0, w - ps)
            image = image[y : y + ps, x : x + ps]
            mask = mask[y : y + ps, x : x + ps]
        else:
            image = np.array(Image.fromarray(image).resize((ps, ps), Image.Resampling.BILINEAR))
            mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((ps, ps), Image.Resampling.NEAREST)) / 255.0

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        # HWC -> CHW, normalize ImageNet
        img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t


def build_model():
    try:
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
        print("Модель: smp.Unet + efficientnet-b0")
        return model
    except ImportError:
        print("segmentation_models_pytorch не найден — простой U-Net на torch")
        return SimpleUNet()


class SimpleUNet(nn.Module):
    """Fallback без smp."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice_loss = 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    dices = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        dices.append(dice_score(logits, masks))
    return total_loss / max(len(loader), 1), float(np.mean(dices)) if dices else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--positive-crop-prob", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("TRAIN U-Net (тальк)")
    print(f"device: {device}", end="")
    if torch.cuda.is_available():
        print(f" — {torch.cuda.get_device_name(0)}")
    else:
        print()

    pairs = load_pairs()
    if len(pairs) < 10:
        print(f"Мало пар image+mask: {len(pairs)}. Сначала: python scripts/batch_analyze.py --mode extract_masks")
        return 1

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"пар image+mask: {len(pairs)} (train {len(train_pairs)}, val {n_val})")

    train_ds = TalcDataset(
        train_pairs,
        args.patch_size,
        augment=True,
        positive_crop_prob=args.positive_crop_prob,
    )
    val_ds = TalcDataset(val_pairs, args.patch_size, augment=False, positive_crop_prob=0.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model().to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    best_path = MODELS_DIR / "best_talc_unet.pth"

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va_loss, va_dice = validate(model, val_loader, criterion, device)
        scheduler.step(va_dice)

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_dice {va_dice:.4f}",
            flush=True,
        )

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": va_dice,
                    "patch_size": args.patch_size,
                    "encoder": "efficientnet-b0",
                },
                best_path,
            )
            print(f"  -> сохранено {best_path} (dice={va_dice:.4f})", flush=True)

    print("=" * 60)
    print(f"Готово за {(time.time()-t0)/60:.1f} мин. Лучший dice: {best_dice:.4f}")
    print(f"Модель: {best_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

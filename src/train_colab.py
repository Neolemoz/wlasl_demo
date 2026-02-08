import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data_kaggle_wlasl import load_wlasl_dataset

FRAMES = 8
SIZE = 112
BATCH_SIZE = 8
LEARNING_RATE = 1e-3


class TinyFrameCNN(nn.Module):
    def __init__(self, in_ch: int = 3, feat_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.flatten(1)


class TinyVideoClassifier(nn.Module):
    def __init__(self, num_classes: int, frames: int, size: int, feat_dim: int = 128) -> None:
        super().__init__()
        self.frames = frames
        self.size = size
        self.backbone = TinyFrameCNN(in_ch=3, feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x)
        feat = feat.view(b, t, -1).mean(dim=1)
        return self.classifier(feat)


def sample_frames(video_path: str, num_frames: int, size: int) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read: {video_path}")

    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])

    arr = np.stack(frames[:num_frames], axis=0)  # [T,H,W,C]
    arr = np.transpose(arr, (0, 3, 1, 2))  # [T,C,H,W]
    return torch.from_numpy(arr).float()


class VideoDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], frames: int, size: int) -> None:
        self.samples = samples
        self.frames = frames
        self.size = size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label_idx = self.samples[idx]
        x = sample_frames(path, self.frames, self.size)
        return x, label_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab baseline trainer.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_classes", type=int, default=100)
    parser.add_argument("--max_samples_per_class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, labels = load_wlasl_dataset(
        args.data_root, args.max_classes, args.max_samples_per_class
    )

    with (out_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f)

    train_ds = VideoDataset(train_samples, FRAMES, SIZE)
    val_ds = VideoDataset(val_samples, FRAMES, SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = TinyVideoClassifier(num_classes=len(labels), frames=FRAMES, size=SIZE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(
        "dataset:",
        f"train={len(train_ds)}",
        f"val={len(val_ds)}",
        f"classes={len(labels)}",
        f"frames={FRAMES}",
        f"size={SIZE}",
    )

    best_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"train {epoch}"):
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"val {epoch}"):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

        train_loss /= max(1, len(train_ds))
        val_loss /= max(1, len(val_ds))
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt = {
                "state_dict": model.state_dict(),
                "meta": {
                    "num_classes": len(labels),
                    "frames": FRAMES,
                    "size": SIZE,
                },
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"saved best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

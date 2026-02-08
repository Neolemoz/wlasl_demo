import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


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
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x)
        feat = feat.view(b, t, -1).mean(dim=1)
        return self.classifier(feat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TorchScript model.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--out_ts", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    labels_path = Path(args.labels)
    out_path = Path(args.out_ts)

    if not ckpt_path.exists():
        raise RuntimeError(f"ckpt not found: {ckpt_path}")
    if not labels_path.exists():
        raise RuntimeError(f"labels not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    num_classes = int(meta.get("num_classes", len(labels)))
    frames = int(meta.get("frames", 8))
    size = int(meta.get("size", 112))

    model = TinyVideoClassifier(num_classes=num_classes, frames=frames, size=size)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    example = torch.zeros(1, frames, 3, size, size)
    scripted = torch.jit.trace(model, example)
    out = scripted(example)
    print(f"TorchScript output shape: {tuple(out.shape)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    print(f"Saved TorchScript: {out_path}")


if __name__ == "__main__":
    main()

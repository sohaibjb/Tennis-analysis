"""Train the ResNet50 court keypoint model (14 points) and save it to weights/keypoints_resnet50.pth.

Dataset: TennisCourtDetector format (github.com/yastrebksv/TennisCourtDetector, 8,841 broadcast frames):
  data/
    images/<id>.png
    data_train.json      [{"id": "...", "kps": [[x, y] * 14]}, ...]   (pixel coords in the original image)
    data_val.json

  python training/train_keypoints.py --data data --epochs 40
  python training/train_keypoints.py --data data --epochs 40 --device cuda:0 --batch 32

The saved model matches the project's loader exactly:
  torchvision resnet50, fc = Linear(2048, 28), input 224x224 RGB, ImageNet-normalised,
  outputs (x0, y0, x1, y1, ...) in pixels of the 224x224 input  -> config: output: pixels
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# index i <-> FLIP[i] when the image is mirrored left/right (see tennis/court.py for the order)
FLIP = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]


class CourtKeypoints(Dataset):
    def __init__(self, root: Path, split: str, augment: bool):
        self.root = Path(root)
        self.items = json.loads((self.root / f"data_{split}.json").read_text())
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(str(self.root / "images" / f"{it['id']}.png"))
        if img is None:
            img = cv2.imread(str(self.root / "images" / f"{it['id']}.jpg"))
        h, w = img.shape[:2]
        kps = np.array(it["kps"], np.float32).reshape(-1, 2)
        if self.augment:
            if random.random() < 0.5:                                   # mirror + swap left/right points
                img = img[:, ::-1]
                kps[:, 0] = w - 1 - kps[:, 0]
                kps = kps[FLIP]
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180  # court colours vary between tournaments
            hsv[..., 1] *= random.uniform(0.7, 1.3)
            hsv[..., 2] *= random.uniform(0.7, 1.3)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        x = cv2.cvtColor(cv2.resize(np.ascontiguousarray(img), (SIZE, SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        x = ((x - MEAN) / STD).transpose(2, 0, 1)
        y = kps * np.array([SIZE / w, SIZE / h], np.float32)
        return torch.from_numpy(x), torch.from_numpy(y.reshape(-1)), torch.tensor([w / SIZE, h / SIZE])


def evaluate(model, loader, device):
    model.eval()
    errs = []
    with torch.inference_mode():
        for x, y, s in loader:
            p = model(x.to(device)).cpu().view(len(x), -1, 2)
            t = y.view(len(x), -1, 2)
            errs.append(((p - t) * s[:, None, :]).norm(dim=2).flatten())   # error in original-image pixels
    e = torch.cat(errs)
    return float(e.mean()), float(e.median())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with images/, data_train.json, data_val.json")
    ap.add_argument("--out", default="weights/keypoints_resnet50.pth")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-imagenet", action="store_true", help="start from random weights (offline tests only)")
    args = ap.parse_args()

    train = DataLoader(CourtKeypoints(args.data, "train", True), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=True)
    val = DataLoader(CourtKeypoints(args.data, "val", False), batch_size=args.batch, num_workers=args.workers)

    weights = None if args.no_imagenet else torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 28)
    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train))
    loss_fn = torch.nn.SmoothL1Loss(beta=2.0)                  # robust to the odd bad label

    best = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for i, (x, y, _) in enumerate(train):
            x, y = x.to(args.device), y.to(args.device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            if i % 50 == 0:
                print(f"epoch {ep + 1}/{args.epochs}  iter {i}/{len(train)}  loss {loss.item():.3f}", flush=True)
        mean_px, med_px = evaluate(model, val, args.device)
        print(f"epoch {ep + 1}: train loss {tot / max(1, len(train)):.3f} | val error mean {mean_px:.1f} px, "
              f"median {med_px:.1f} px (original resolution)")
        if mean_px < best:
            best = mean_px
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out}")
    print(f"Done. Best mean keypoint error {best:.1f} px. config.yaml: input_size 224, output pixels.")


if __name__ == "__main__":
    main()

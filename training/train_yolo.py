"""Train (fine-tune) a YOLO26 detector for players or for the ball, and copy the result into weights/.

Dataset: YOLO format with a data.yaml (Roboflow "Download dataset -> YOLOv8/YOLO11 format" works as is).

  # ball  (small object: train at a larger image size)
  python training/train_yolo.py --data datasets/tennis-ball/data.yaml --out weights/ball_yolo26.pt --imgsz 1280 --epochs 100

  # players
  python training/train_yolo.py --data datasets/tennis-players/data.yaml --out weights/players_yolo26.pt --imgsz 640 --epochs 60

Start from yolo26s.pt (default) or yolo26m.pt on a good GPU; yolo26n.pt for CPU / quick tests.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--out", required=True, help="where to copy best.pt, e.g. weights/ball_yolo26.pt")
    ap.add_argument("--model", default="yolo26s.pt", help="starting weights (auto-downloaded)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto batch size for your GPU")
    ap.add_argument("--device", default=None, help="0, 0,1, cpu, mps")
    ap.add_argument("--patience", type=int, default=25, help="stop early after this many epochs without improvement")
    args = ap.parse_args()

    from ultralytics import YOLO
    name = Path(args.out).stem
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                patience=args.patience, project="runs/train", name=name, exist_ok=True,
                # tennis-friendly augmentation: no vertical flips (court perspective), mild scale changes
                flipud=0.0, fliplr=0.5, scale=0.3, mosaic=1.0, close_mosaic=10)
    best = Path(model.trainer.best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"\nSaved {args.out}   mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")
    print("Check the class names in config.yaml (`classes: null` uses every class).")


if __name__ == "__main__":
    main()

"""Diagnose ball detection on your video: which model is used, and which settings find the ball.

  python check_ball.py --source input_videos/vid1.mp4

Samples frames across the video, runs the ball model at several image sizes and confidence
thresholds, prints how often a ball is found, and saves annotated sample frames to
runs/ball_check/ so you can see what the model picked.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", type=int, default=40, help="number of frames to sample")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    from tennis.detectors import _class_ids, resolve_device

    cfg = yaml.safe_load(Path(args.config).read_text())
    b = cfg["models"]["ball"]
    device = resolve_device(args.device or cfg.get("device", "auto"))
    if Path(b["weights"]).exists():
        weights, classes = b["weights"], b.get("classes")
        print(f"Ball model: YOUR weights {b['weights']}")
    else:
        weights, classes = b.get("fallback_weights", "yolo26s.pt"), b.get("fallback_classes")
        print(f"Ball model: {b['weights']} NOT FOUND -> public fallback {weights} {classes}.")
        print("  The public model often misses tennis balls. Put a trained ball model at that path for real results.")
    model = YOLO(weights)
    names = list(model.names.values())
    print(f"  classes in this model ({len(names)}): {names if len(names) <= 10 else names[:10] + ['...']}")
    cls_ids = _class_ids(model, classes)
    print(f"  using classes: {classes or 'all'} -> ids {cls_ids or 'all'}")

    cap = cv2.VideoCapture(args.source)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h}, {n} frames. Sampling {args.frames} frames.")
    frames = []
    for i in np.linspace(0, max(0, n - 1), args.frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append((int(i), f))
    cap.release()
    if not frames:
        raise SystemExit("Could not read frames from the video.")

    sizes = sorted({640, 960, 1280, int(round(max(w, h) / 32) * 32)})
    confs = [0.05, 0.10, 0.15, 0.25]
    print("\n% of sampled frames with a ball found (rows: imgsz, columns: conf)")
    print("imgsz  " + "  ".join(f"{c:>5.2f}" for c in confs) + "   best conf seen")
    best = (0, None, None)
    for s in sizes:
        hits = {c: 0 for c in confs}
        top = []
        for _, f in frames:
            r = model.predict(f, conf=min(confs), imgsz=s, classes=cls_ids, device=device, verbose=False)[0]
            cs = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.array([])
            top.append(cs.max() if len(cs) else 0.0)
            for c in confs:
                hits[c] += int((cs >= c).any())
        row = [100 * hits[c] / len(frames) for c in confs]
        print(f"{s:>5}  " + "  ".join(f"{v:5.0f}" for v in row) + f"   {max(top):.2f}")
        for c, v in zip(confs, row):
            if v > best[0] or (v == best[0] and best[2] is not None and c > best[2]):
                best = (v, s, c)

    out = Path("runs/ball_check")
    out.mkdir(parents=True, exist_ok=True)
    s, c = best[1] or 1280, best[2] or 0.1
    for i, f in frames[:: max(1, len(frames) // 8)]:
        r = model.predict(f, conf=c, imgsz=s, classes=cls_ids, device=device, verbose=False)[0]
        img = f.copy()
        for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 255, 255), 2)
            cv2.putText(img, f"{cf:.2f}", (x2 + 8, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"frame_{i:05d}.jpg"), img)

    print(f"\nSample frames with detections saved to {out}/  (imgsz={s}, conf={c})")
    if best[0] == 0:
        print("\nNo ball found in any sampled frame. This model can't see the ball in this video:")
        print("  - use a ball model trained on tennis (see weights/README.txt and training/README.md)")
        print("  - check the ball is visible at all at this resolution (very small / blurred balls need training data like your video)")
    elif best[0] < 50:
        print(f"\nBest setting finds the ball in only {best[0]:.0f}% of frames (gaps up to ball_max_gap frames are filled).")
        print(f"Set in config.yaml under ball:  imgsz: {s}   conf: {c}   and consider a better-trained ball model.")
    else:
        print(f"\nGood. Set in config.yaml under ball:  imgsz: {s}   conf: {c}")


if __name__ == "__main__":
    main()

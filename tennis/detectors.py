"""Wrappers around the three trained models.

  PlayerDetector    Ultralytics YOLO26 + ByteTrack/BoT-SORT  -> boxes with track IDs
  BallDetector      Ultralytics YOLO26                       -> best ball box per frame
  CourtKeypointModel ResNet50 regression head (28 outputs)    -> 14 (x, y) image points
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def resolve_device(device: str | None) -> str:
    if device and device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Detection:
    box: np.ndarray                     # x1, y1, x2, y2 in image px
    conf: float
    cls: int = 0
    track_id: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def foot(self) -> np.ndarray:       # bottom-centre: where the player touches the ground
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2, y2], dtype=np.float32)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


def _fp16(on: bool) -> dict:
    """Only pass `half` when FP16 is wanted (newer Ultralytics versions warn about the argument)."""
    return {"half": True} if on else {}


def _class_ids(model, names):
    """Turn a list of class names (or ids) into the model's class ids. None = all classes."""
    if not names:
        return None
    lookup = {v.lower(): k for k, v in model.names.items()}
    ids = []
    for n in names:
        if isinstance(n, int):
            ids.append(n)
        elif n.lower() in lookup:
            ids.append(lookup[n.lower()])
        else:
            raise ValueError(f"Class '{n}' not in model classes {list(model.names.values())}")
    return ids


class PlayerDetector:
    """YOLO26 player detector with Ultralytics' built-in multi-object tracker."""

    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.35, imgsz: int = 640,
                 classes=None, tracker: str = "bytetrack.yaml", half: bool = False):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device, self.conf, self.imgsz, self.half = device, conf, imgsz, half
        self.tracker = tracker
        self.classes = _class_ids(self.model, classes)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.track(frame, persist=True, conf=self.conf, imgsz=self.imgsz, classes=self.classes,
                             tracker=self.tracker, device=self.device, verbose=False, **_fp16(self.half))[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()
        clss = b.cls.cpu().numpy().astype(int)
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else [None] * len(xyxy)
        return [Detection(xyxy[i], float(confs[i]), int(clss[i]), None if ids[i] is None else int(ids[i]))
                for i in range(len(xyxy))]

    def reset(self) -> None:
        """Clear tracker state (call after a camera cut)."""
        pred = getattr(self.model, "predictor", None)
        for t in getattr(pred, "trackers", None) or []:
            try:
                t.reset()
            except Exception:
                pass


class BallDetector:
    """YOLO26 ball detector. Returns the most confident ball (or None)."""

    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.15, imgsz: int = 1280,
                 classes=None, half: bool = False):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device, self.conf, self.imgsz, self.half = device, conf, imgsz, half
        self.classes = _class_ids(self.model, classes)

    def __call__(self, frame: np.ndarray, prefer_near: np.ndarray | None = None,
                 max_dist_px: float | None = None) -> Detection | None:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, classes=self.classes,
                               device=self.device, verbose=False, **_fp16(self.half))[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return None
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()
        score = confs.copy()
        if prefer_near is not None and max_dist_px:
            # Several candidates (e.g. a spare ball in a pocket): favour the one near the track
            c = (xyxy[:, :2] + xyxy[:, 2:]) / 2
            d = np.linalg.norm(c - prefer_near, axis=1)
            score = confs - 0.5 * np.clip(d / max_dist_px, 0, 1)
        i = int(np.argmax(score))
        return Detection(xyxy[i], float(confs[i]), int(b.cls[i]))


class CourtKeypointModel:
    """ResNet50 regressing 14 court keypoints (28 numbers) from the full frame.

    Accepts either a pickled nn.Module or a state_dict for torchvision's resnet50 with
    `fc = Linear(2048, 28)`. The input is the whole frame resized to `input_size`,
    RGB, ImageNet-normalised. `output` says how the 28 numbers are expressed:
      "pixels"      -> coordinates in the resized input image (default)
      "normalized"  -> coordinates in 0..1
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, weights: str, device: str = "cpu", input_size: int = 224, num_keypoints: int = 14,
                 output: str = "pixels", normalize: bool = True):
        import torch
        import torchvision
        self.torch = torch
        self.device, self.size, self.n = device, int(input_size), int(num_keypoints)
        self.output, self.normalize = output, normalize

        obj = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            model = obj
        else:
            sd = obj
            if isinstance(obj, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in obj and isinstance(obj[key], dict):
                        sd = obj[key]
                        break
            sd = {k.removeprefix("module.").removeprefix("model."): v for k, v in sd.items()}
            model = torchvision.models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, self.n * 2)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                raise RuntimeError(
                    f"Keypoint weights don't match ResNet50 with fc->{self.n * 2}. Missing keys: {missing[:6]}... "
                    "If your head differs, load your own nn.Module and save it with torch.save(model, path).")
            if unexpected:
                print(f"[keypoints] ignoring {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
        self.model = model.eval().to(device)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame, (self.size, self.size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.normalize:
            img = (img - self.MEAN) / self.STD
        t = self.torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            out = self.model(t)[0].float().cpu().numpy().reshape(-1, 2)[: self.n]
        if self.output == "normalized":
            out = out * np.array([w, h], dtype=np.float32)
        else:
            out = out * np.array([w / self.size, h / self.size], dtype=np.float32)
        return out.astype(np.float32)


class ManualCourt:
    """Court keypoints from 4 corners clicked once (see calibrate_court.py).

    Replaces the keypoint model for a fixed camera: the 4 doubles corners (far-left,
    far-right, near-left, near-right) define the homography, and all 14 points are
    projected from it every frame.
    """

    def __init__(self, corners_px):
        from .court import COURT_KEYPOINTS_M
        c = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
        H = cv2.getPerspectiveTransform(COURT_KEYPOINTS_M[:4].copy(), c)          # court m -> image px
        self.kps = cv2.perspectiveTransform(COURT_KEYPOINTS_M.reshape(-1, 1, 2), H).reshape(-1, 2)

    @classmethod
    def from_json(cls, path):
        import json
        with open(path) as f:
            return cls(json.load(f)["corners_px"])

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.kps.copy()

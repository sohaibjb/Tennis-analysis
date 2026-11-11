"""Court geometry and image <-> court mapping.

Court coordinates are in metres with the origin at the centre of the net:
  x  across the court (left -> right as seen from the main camera)
  y  along the court (negative = far half, positive = near half)

The 14 reference keypoints follow the ordering used by the common
TennisCourtDetector dataset (and most ResNet50 keypoint models trained on it):

   0 ---- 4 -------------- 6 ---- 1        far baseline
          |                |
          8 ------12 ----- 9               far service line
          |       |        |
   =======|=======|========|=======        net
          |       |        |
          10 -----13 ----- 11              near service line
          |                |
   2 ---- 5 -------------- 7 ---- 3        near baseline

 0-3   doubles corners          (far-left, far-right, near-left, near-right)
 4,5   left singles sideline    (far end, near end)
 6,7   right singles sideline   (far end, near end)
 8,9   far service line ends    10,11 near service line ends
 12,13 centre service line ends (far, near)

If your model was trained with a different ordering, edit COURT_KEYPOINTS_M
(or set `keypoints.order` in config.yaml) so index i matches your model's output i.
"""
from __future__ import annotations

import cv2
import numpy as np

HALF_LENGTH = 23.77 / 2          # 11.885 m
HALF_WIDTH = 10.97 / 2           # 5.485 m (doubles)
SINGLES_HALF_WIDTH = 8.23 / 2    # 4.115 m
SERVICE_LINE = 6.40              # distance from net to service line
NET_POST_X = HALF_WIDTH + 0.914

COURT_KEYPOINTS_M = np.array([
    [-HALF_WIDTH, -HALF_LENGTH], [HALF_WIDTH, -HALF_LENGTH],
    [-HALF_WIDTH, HALF_LENGTH], [HALF_WIDTH, HALF_LENGTH],
    [-SINGLES_HALF_WIDTH, -HALF_LENGTH], [-SINGLES_HALF_WIDTH, HALF_LENGTH],
    [SINGLES_HALF_WIDTH, -HALF_LENGTH], [SINGLES_HALF_WIDTH, HALF_LENGTH],
    [-SINGLES_HALF_WIDTH, -SERVICE_LINE], [SINGLES_HALF_WIDTH, -SERVICE_LINE],
    [-SINGLES_HALF_WIDTH, SERVICE_LINE], [SINGLES_HALF_WIDTH, SERVICE_LINE],
    [0.0, -SERVICE_LINE], [0.0, SERVICE_LINE],
], dtype=np.float32)

# Line segments (in metres) used to draw the court
COURT_LINES_M = []
for s in (-1, 1):
    COURT_LINES_M += [
        ((-HALF_WIDTH, s * HALF_LENGTH), (HALF_WIDTH, s * HALF_LENGTH)),            # baselines
        ((s * HALF_WIDTH, -HALF_LENGTH), (s * HALF_WIDTH, HALF_LENGTH)),            # doubles sidelines
        ((s * SINGLES_HALF_WIDTH, -HALF_LENGTH), (s * SINGLES_HALF_WIDTH, HALF_LENGTH)),  # singles sidelines
        ((-SINGLES_HALF_WIDTH, s * SERVICE_LINE), (SINGLES_HALF_WIDTH, s * SERVICE_LINE)),  # service lines
        ((0.0, s * HALF_LENGTH), (0.0, s * (HALF_LENGTH - 0.15))),                  # centre marks
    ]
COURT_LINES_M.append(((0.0, -SERVICE_LINE), (0.0, SERVICE_LINE)))                   # centre service line
NET_LINE_M = ((-NET_POST_X, 0.0), (NET_POST_X, 0.0))


class CourtMapper:
    """Fits an image -> court homography from detected keypoints (RANSAC)."""

    def __init__(self, reference: np.ndarray = COURT_KEYPOINTS_M, ransac_thresh_m: float = 0.5,
                 min_points: int = 6, max_residual_cm: float = 60.0):
        self.reference = np.asarray(reference, dtype=np.float32)
        self.ransac_thresh_m = ransac_thresh_m
        self.min_points = min_points
        self.max_residual_cm = max_residual_cm
        self.H: np.ndarray | None = None        # image px -> court m
        self.H_inv: np.ndarray | None = None    # court m -> image px
        self.residual_cm: float = float("nan")
        self.inliers: int = 0

    @property
    def ready(self) -> bool:
        return self.H is not None

    def reset(self) -> None:
        self.H = self.H_inv = None
        self.residual_cm = float("nan")
        self.inliers = 0

    def update(self, keypoints_px: np.ndarray, valid: np.ndarray | None = None) -> bool:
        """Re-fit the homography. Keeps the previous fit if the new one is unusable."""
        kp = np.asarray(keypoints_px, dtype=np.float32)
        ok = np.isfinite(kp).all(axis=1)
        if valid is not None:
            ok &= np.asarray(valid, dtype=bool)
        if ok.sum() < self.min_points:
            return False
        src, dst = kp[ok], self.reference[ok]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh_m)
        if H is None:
            return False
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        inl = mask.ravel().astype(bool)
        if inl.sum() < self.min_points:
            return False
        residual = float(np.sqrt(np.mean(np.sum((proj[inl] - dst[inl]) ** 2, axis=1))) * 100)
        if residual > self.max_residual_cm:
            return False
        self.H, self.H_inv = H, np.linalg.inv(H)
        self.residual_cm, self.inliers = residual, int(inl.sum())
        return True

    def to_court(self, pts_px) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts_m) -> np.ndarray:
        pts = np.asarray(pts_m, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

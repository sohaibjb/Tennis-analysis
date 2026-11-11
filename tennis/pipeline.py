"""Streaming pipeline: one frame in, annotated frames out.

Works the same for a video file and a live stream. Frames are held in a short
look-back buffer (`lag_frames`) so that missed ball detections can be filled in
by interpolation before the frame is drawn. Output latency = lag_frames / fps.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .analysis import (FAR, NEAR, BallTracker, FrameRecord, PlayerMotion, PlayerSelector, ShotDetector)
from .court import COURT_KEYPOINTS_M, CourtMapper


class TennisPipeline:
    def __init__(self, player_detector, ball_detector, keypoint_model, cfg: dict, fps: float, annotator=None):
        self.players_m = player_detector
        self.ball_m = ball_detector
        self.kp_m = keypoint_model
        self.fps = fps
        self.annotator = annotator

        kc = cfg.get("keypoints", {})
        pc = cfg.get("pipeline", {})
        ac = cfg.get("analysis", {})
        order = kc.get("order")
        ref = COURT_KEYPOINTS_M[order] if order else COURT_KEYPOINTS_M
        self.kp_every = int(kc.get("every_n_frames", 1))
        self.kp_alpha = float(kc.get("smoothing", 0.5))
        self.cut_px = float(pc.get("cut_threshold_px", 40))
        self.court_lost_frames = int(pc.get("court_lost_frames", 15))
        self.lag = int(pc.get("lag_frames", 10))

        self.mapper = CourtMapper(ref, ransac_thresh_m=float(kc.get("ransac_thresh_m", 0.5)),
                                  max_residual_cm=float(kc.get("max_residual_cm", 60)))
        self.selector = PlayerSelector()
        self.ball = BallTracker(max_gap=int(pc.get("ball_max_gap", 10)),
                                max_jump_px=float(pc.get("ball_max_jump_px", 150)))
        self.motion = {s: PlayerMotion(smoothing=float(ac.get("position_smoothing", 0.35)),
                                       speed_window_s=float(ac.get("speed_window_s", 0.17)),
                                       distance_step_m=float(ac.get("distance_step_m", 0.25)))
                       for s in (NEAR, FAR)}
        self.shots = ShotDetector(fps, min_frames=int(ac.get("hit_min_frames", 4)),
                                  min_gap_s=float(ac.get("hit_min_gap_s", 0.45)),
                                  reach=float(ac.get("hit_reach", 1.2)),
                                  min_v_court_m=float(ac.get("hit_min_ball_speed_m", 0.15)) * 30.0 / fps,
                                  max_shot_s=float(ac.get("max_shot_s", 3.5)))

        self.buffer: deque[FrameRecord] = deque()
        self.kp_smooth: np.ndarray | None = None
        self.court_fail = 0
        self.idx = 0

    # ------------------------------------------------------------------ per frame
    def process(self, frame: np.ndarray, t: float | None = None) -> list[FrameRecord]:
        t = self.idx / self.fps if t is None else t
        rec = FrameRecord(idx=self.idx, t=t, frame=frame)
        self.idx += 1

        # 1) court keypoints -> homography
        if rec.idx % self.kp_every == 0 or self.kp_smooth is None:
            t0 = time.perf_counter()
            kp = self.kp_m(frame)
            rec.timings_ms["keypoints"] = (time.perf_counter() - t0) * 1000
            if self.kp_smooth is not None and np.median(np.linalg.norm(kp - self.kp_smooth, axis=1)) > self.cut_px:
                rec.cut = True
                self._on_cut()
            self.kp_smooth = kp if self.kp_smooth is None else self.kp_smooth + self.kp_alpha * (kp - self.kp_smooth)
            h, w = frame.shape[:2]
            inside = ((self.kp_smooth[:, 0] > -0.05 * w) & (self.kp_smooth[:, 0] < 1.05 * w) &
                      (self.kp_smooth[:, 1] > -0.05 * h) & (self.kp_smooth[:, 1] < 1.05 * h))
            if self.mapper.update(self.kp_smooth, inside):
                self.court_fail = 0
            else:
                self.court_fail += 1
                if self.court_fail > self.court_lost_frames:
                    self.mapper.reset()                 # court not visible (close-up / replay)
        rec.keypoints = None if self.kp_smooth is None else self.kp_smooth.copy()
        rec.H = None if self.mapper.H is None else self.mapper.H.copy()
        rec.court_residual_cm = self.mapper.residual_cm

        # 2) players
        t0 = time.perf_counter()
        rec.all_players = self.players_m(frame)
        rec.timings_ms["players"] = (time.perf_counter() - t0) * 1000
        rec.players, rec.player_court = self.selector.select(rec.all_players, self.mapper, frame.shape[0])

        # 3) ball
        t0 = time.perf_counter()
        det = self.ball_m(frame, prefer_near=self.ball.last_center, max_dist_px=self.ball.max_jump_px * 2)
        rec.timings_ms["ball"] = (time.perf_counter() - t0) * 1000
        self.ball.add(rec, det, self.buffer)

        self.buffer.append(rec)
        out = []
        while len(self.buffer) > self.lag:
            out.append(self._finalize(self.buffer.popleft()))
        return out

    def flush(self) -> list[FrameRecord]:
        out = [self._finalize(r) for r in list(self.buffer)]
        self.buffer.clear()
        self.shots.flush()
        if out:
            out[-1].hits += self.shots.events
        return out

    # ------------------------------------------------------------------ internals
    def _on_cut(self):
        self.kp_smooth = None
        self.mapper.reset()
        self.selector.reset()
        self.ball.reset()
        if hasattr(self.players_m, "reset"):
            self.players_m.reset()

    def _finalize(self, rec: FrameRecord) -> FrameRecord:
        if rec.cut:
            for m in self.motion.values():
                m.reset_track()
            self.shots.reset()
        if rec.ball is not None and rec.H is not None:
            import cv2
            rec.ball.court = cv2.perspectiveTransform(rec.ball.center.reshape(1, 1, 2).astype(np.float32),
                                                      rec.H).reshape(2)
        for side, m in self.motion.items():
            rec.speed_kmh[side] = m.update(rec.t, rec.player_court.get(side))
        rec.hits = list(self.shots.update(rec))
        if self.annotator is not None:
            rec.annotated = self.annotator.draw(rec, self)
        return rec

    # ------------------------------------------------------------------ results
    def player_stats(self, side: int) -> dict:
        m = self.motion[side]
        hits = [h for h in self.shots.hits if h.side == side]
        speeds = [h.speed_kmh for h in hits if h.speed_kmh is not None]
        return {
            "speed_kmh": round(m.speed_kmh, 2),
            "top_speed_kmh": round(m.top_kmh, 2),
            "avg_speed_moving_kmh": round(m.avg_moving_kmh, 2),
            "distance_m": round(m.distance_m, 2),
            "shots": len(hits),
            "avg_shot_speed_kmh": round(float(np.mean(speeds)), 1) if speeds else None,
            "max_shot_speed_kmh": round(float(np.max(speeds)), 1) if speeds else None,
            "last_shot_speed_kmh": round(speeds[-1], 1) if speeds else None,
        }

    def summary(self) -> dict:
        return {
            "frames": self.idx,
            "fps": self.fps,
            "players": {"P1_near": self.player_stats(NEAR), "P2_far": self.player_stats(FAR)},
            "shots": [{"frame": h.idx, "time_s": round(h.t, 3), "player": "P1" if h.side == NEAR else "P2",
                       "speed_kmh": None if h.speed_kmh is None else round(h.speed_kmh, 1),
                       "player_court_xy_m": None if h.player_court is None else [round(float(v), 3) for v in h.player_court]}
                      for h in self.shots.hits],
        }

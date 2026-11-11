"""Tracking logic and metrics that sit on top of the raw model outputs.

  PlayerSelector  picks the two players out of all person boxes (ignores umpire, ball kids, crowd)
  BallTracker     rejects outlier ball detections and fills short gaps by interpolation
  PlayerMotion    court-space position smoothing, speed and distance per player
  ShotDetector    finds racket contacts from ball direction changes near a player, and
                  turns consecutive hits into shot speeds
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .court import HALF_LENGTH, HALF_WIDTH, CourtMapper
from .detectors import Detection

NEAR, FAR = 0, 1           # side index: 0 = near half (bottom of the image), 1 = far half
SIDE_NAMES = {NEAR: "P1", FAR: "P2"}


# --------------------------------------------------------------------------- data records
@dataclass
class BallState:
    center: np.ndarray
    box: np.ndarray
    conf: float
    interpolated: bool = False
    court: np.ndarray | None = None      # ground-plane projection (exact only when the ball is on the ground)


@dataclass
class FrameRecord:
    idx: int
    t: float
    frame: np.ndarray | None
    keypoints: np.ndarray | None = None
    H: np.ndarray | None = None
    court_residual_cm: float = float("nan")
    cut: bool = False
    all_players: list = field(default_factory=list)
    players: dict = field(default_factory=dict)        # side -> Detection | None
    player_court: dict = field(default_factory=dict)   # side -> np.array([x, y]) metres | None
    ball: BallState | None = None
    speed_kmh: dict = field(default_factory=dict)      # side -> float
    hits: list = field(default_factory=list)           # hits confirmed while finalizing this frame
    timings_ms: dict = field(default_factory=dict)
    annotated: np.ndarray | None = None


# --------------------------------------------------------------------------- players
class PlayerSelector:
    """Keeps one tracked person per half of the court."""

    def __init__(self, max_outside_x: float = 4.0, max_behind_baseline: float = 8.0, lost_frames: int = 15):
        self.max_outside_x = max_outside_x
        self.max_behind = max_behind_baseline
        self.lost_frames = lost_frames
        self.locked: dict[int, int | None] = {NEAR: None, FAR: None}
        self.missing: dict[int, int] = {NEAR: 0, FAR: 0}

    def reset(self):
        self.locked = {NEAR: None, FAR: None}
        self.missing = {NEAR: 0, FAR: 0}

    def select(self, dets: list[Detection], mapper: CourtMapper, frame_h: int) -> tuple[dict, dict]:
        chosen: dict[int, Detection | None] = {NEAR: None, FAR: None}
        court: dict[int, np.ndarray | None] = {NEAR: None, FAR: None}
        if not dets:
            return chosen, court

        if not mapper.ready:
            # No court fit (close-up, replay...). Fall back to the two biggest people, split by image height.
            big = sorted(dets, key=lambda d: d.height, reverse=True)[:2]
            big.sort(key=lambda d: d.foot[1], reverse=True)
            if big:
                chosen[NEAR] = big[0]
            if len(big) > 1:
                chosen[FAR] = big[1]
            return chosen, court

        feet = mapper.to_court(np.stack([d.foot for d in dets]))
        for side, sign in ((NEAR, 1.0), (FAR, -1.0)):
            cands = []
            for d, (x, y) in zip(dets, feet):
                if sign * y < -0.5:                      # wrong half
                    continue
                if abs(x) > HALF_WIDTH + self.max_outside_x or abs(y) > HALF_LENGTH + self.max_behind:
                    continue
                cands.append((d, np.array([x, y], dtype=np.float32)))
            pick = None
            if self.locked[side] is not None:
                pick = next((c for c in cands if c[0].track_id == self.locked[side]), None)
            if pick is None and cands:
                target = np.array([0.0, sign * (HALF_LENGTH - 1.5)], dtype=np.float32)
                pick = min(cands, key=lambda c: np.linalg.norm(c[1] - target) - 3.0 * c[0].conf)
            if pick is not None:
                chosen[side], court[side] = pick
                self.missing[side] = 0
                if self.locked[side] != pick[0].track_id:
                    self.locked[side] = pick[0].track_id
            else:
                self.missing[side] += 1
                if self.missing[side] > self.lost_frames:
                    self.locked[side] = None
        return chosen, court


# --------------------------------------------------------------------------- ball
class BallTracker:
    """Outlier rejection + linear gap filling inside the pipeline's look-back buffer."""

    def __init__(self, max_gap: int = 10, max_jump_px: float = 150.0):
        self.max_gap = max_gap
        self.max_jump_px = max_jump_px
        self.last_idx: int | None = None
        self.last_c: np.ndarray | None = None
        self.last_box_wh = np.array([10.0, 10.0], dtype=np.float32)
        self.rejects = 0

    def reset(self):
        self.last_idx = self.last_c = None
        self.rejects = 0

    @property
    def last_center(self):
        return self.last_c

    def add(self, rec: FrameRecord, det: Detection | None, buffer) -> None:
        rec.ball = None
        if det is None:
            return
        c = det.center
        if self.last_idx is not None:
            gap = rec.idx - self.last_idx
            if gap > self.max_gap + 1:
                self.reset()
            elif np.linalg.norm(c - self.last_c) > self.max_jump_px * gap and self.rejects < 2:
                self.rejects += 1               # implausible jump: probably a false positive
                return
        self.rejects = 0
        rec.ball = BallState(center=c, box=det.box.copy(), conf=det.conf)
        wh = det.box[2:] - det.box[:2]
        if self.last_idx is not None and rec.idx - self.last_idx > 1:
            a_idx, a_c = self.last_idx, self.last_c
            for r in buffer:
                if a_idx < r.idx < rec.idx and r.ball is None:
                    f = (r.idx - a_idx) / (rec.idx - a_idx)
                    cc = a_c + (c - a_c) * f
                    half = (self.last_box_wh + (wh - self.last_box_wh) * f) / 2
                    r.ball = BallState(center=cc, box=np.r_[cc - half, cc + half], conf=0.0, interpolated=True)
        self.last_idx, self.last_c, self.last_box_wh = rec.idx, c, wh.astype(np.float32)


# --------------------------------------------------------------------------- player motion
class PlayerMotion:
    """Smoothed court position, instantaneous speed, top / average speed and distance for one player."""

    def __init__(self, smoothing: float = 0.35, speed_window_s: float = 0.17, distance_step_m: float = 0.25,
                 max_speed_kmh: float = 40.0, moving_kmh: float = 1.5):
        self.alpha = smoothing
        self.window = speed_window_s
        self.step = distance_step_m
        self.max_speed = max_speed_kmh
        self.moving = moving_kmh
        self.distance_m = 0.0
        self.top_kmh = 0.0
        self._sum = 0.0
        self._n = 0
        self.reset_track()

    def reset_track(self):
        self.pos: np.ndarray | None = None
        self.anchor: np.ndarray | None = None
        self.hist: deque = deque(maxlen=120)
        self.speed_kmh = 0.0

    @property
    def avg_moving_kmh(self) -> float:
        return self._sum / self._n if self._n else 0.0

    def update(self, t: float, court_xy: np.ndarray | None) -> float:
        if court_xy is None:
            self.speed_kmh *= 0.8
            return self.speed_kmh
        prev = self.pos
        self.pos = court_xy.copy() if prev is None else prev + self.alpha * (court_xy - prev)
        # distance: only count movement once the player has left a small radius around the last
        # anchor point, so box jitter (worse on the far side) does not add phantom metres
        if self.anchor is None:
            self.anchor = self.pos.copy()
        else:
            step = float(np.linalg.norm(self.pos - self.anchor))
            if step >= self.step:
                self.distance_m += step
                self.anchor = self.pos.copy()
        self.hist.append((t, self.pos.copy()))
        old = next((h for h in self.hist if t - h[0] <= self.window), None)
        if old is not None and t - old[0] > 1e-6:
            v = float(np.linalg.norm(self.pos - old[1]) / (t - old[0]) * 3.6)
            if v > self.max_speed:          # tracking glitch (ID switch, bad homography): restart the track
                self.reset_track()
                return 0.0
            self.speed_kmh = 0.6 * self.speed_kmh + 0.4 * v
        self.top_kmh = max(self.top_kmh, self.speed_kmh)
        if self.speed_kmh > self.moving:
            self._sum += self.speed_kmh
            self._n += 1
        return self.speed_kmh


# --------------------------------------------------------------------------- shots
@dataclass
class Hit:
    idx: int
    t: float
    side: int
    player_court: np.ndarray | None
    ball_px: np.ndarray
    speed_kmh: float | None = None      # speed of the shot this hit starts (known at the next hit)
    closed: "Hit | None" = None         # previous hit whose shot speed was completed by this hit


class ShotDetector:
    """Detects racket contacts and estimates shot speed.

    A hit is a turning point of the ball's position along the court (ground-plane track from the
    homography; image y if there is no court fit): a local extremum with a clear trend before and
    a fast trend after, within reach of the player on the side the ball is leaving.
    Shot speed = court distance between the hitter (at contact) and the next hitter
    (at their contact) divided by the time between the two contacts.
    """

    def __init__(self, fps: float, min_frames: int = 4, min_gap_s: float = 0.45, reach: float = 1.2,
                 max_shot_s: float = 3.5, min_vy_px: float = 1.5,
                 min_v_court_m: float = 0.15):
        self.min_frames = min_frames
        self.min_gap = max(1, int(min_gap_s * fps))
        self.reach = reach
        self.max_shot_s = max_shot_s
        self.min_vy = min_vy_px
        self.min_v_court = min_v_court_m      # metres per frame along the court
        self.hist: deque = deque(maxlen=2 * (min_frames + 2) + 1)
        self.hits: list[Hit] = []
        self.events: list[Hit] = []     # hits committed on the latest update (for logging / UI)

    def reset(self):
        self.hist.clear()

    def _commit(self, hit: Hit):
        prev = self.hits[-1] if self.hits else None
        if (prev is not None and prev.side != hit.side and hit.t - prev.t <= self.max_shot_s
                and prev.player_court is not None and hit.player_court is not None):
            prev.speed_kmh = float(np.linalg.norm(hit.player_court - prev.player_court) / (hit.t - prev.t) * 3.6)
            hit.closed = prev
        self.hits.append(hit)
        self.events.append(hit)

    def update(self, rec: FrameRecord) -> list[Hit]:
        self.events = []
        if rec.ball is None:
            self.hist.clear()
        else:
            self.hist.append(rec)
        w = self.min_frames + 2
        if len(self.hist) >= 2 * w + 1:
            recs = list(self.hist)[-(2 * w + 1):]
            # Use the ball's ground-plane track along the court (y): travel between the players dominates
            # it, so a turning point means a racket contact, while bounces and toss arcs barely move it.
            use_court = all(r.ball.court is not None for r in recs)
            ys = np.array([r.ball.court[1] if use_court else r.ball.center[1] for r in recs], dtype=np.float64)
            thr = self.min_v_court if use_court else self.min_vy
            c = w
            is_ext = (np.argmax(ys) == c) or (np.argmin(ys) == c)
            if is_ext:
                x = np.arange(w + 1)
                mb = np.polyfit(x, ys[:c + 1], 1)[0]          # slope before the turn (per frame)
                ma = np.polyfit(x, ys[c:], 1)[0]              # slope after the turn
                # the ball leaves the racket fast; before contact it can be slow (serve toss, drop shot)
                if abs(ma) > thr and abs(mb) > 0.3 * thr and np.sign(mb) != np.sign(ma):
                    rk = recs[c]
                    side = NEAR if ma < 0 else FAR            # after contact the ball heads to the other half
                    if self._in_reach(rk, side):
                        self._candidate(Hit(rk.idx, rk.t, side, rk.player_court.get(side), rk.ball.center.copy()))
        return self.events

    def _in_reach(self, rec: FrameRecord, side: int) -> bool:
        d = rec.players.get(side)
        if d is None:
            return False
        x1, y1, x2, y2 = d.box
        m = self.reach * d.height
        b = rec.ball.center
        return x1 - m <= b[0] <= x2 + m and y1 - m <= b[1] <= y2 + 0.3 * m

    def _candidate(self, hit: Hit):
        last = self.hits[-1] if self.hits else None
        if last is not None and last.side == hit.side and hit.idx - last.idx < self.min_gap:
            # same player again within a fraction of a second (e.g. toss apex, then the serve contact):
            # keep the later contact
            self.hits.pop()
            if self.hits:
                self.hits[-1].speed_kmh = None
        self._commit(hit)

    def flush(self):
        pass

"""Annotation layer: boxes, ball trail, keypoints, mini court and stats panel (OpenCV, BGR)."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .analysis import FAR, NEAR
from .court import COURT_KEYPOINTS_M, COURT_LINES_M, HALF_LENGTH, HALF_WIDTH, NET_LINE_M

COL = {
    NEAR: (60, 110, 255),     # P1 orange-red
    FAR: (220, 210, 45),      # P2 cyan
    "ball": (60, 240, 230),   # optic yellow
    "kp": (32, 176, 255),     # amber
    "text": (245, 245, 245),
    "panel": (24, 18, 12),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale, color=COL["text"], thick=1, bg=None, pad=4):
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return w, h


def _blend_rect(img, p1, p2, color, alpha):
    x1, y1 = max(0, int(p1[0])), max(0, int(p1[1]))
    x2, y2 = min(img.shape[1], int(p2[0])), min(img.shape[0], int(p2[1]))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


class MiniCourt:
    def __init__(self, width_px: int, margin_m=(1.8, 3.0)):
        self.mx, self.my = margin_m
        self.scale = width_px / (2 * (HALF_WIDTH + self.mx))
        self.w = int(round(width_px))
        self.h = int(round(2 * (HALF_LENGTH + self.my) * self.scale))
        base = np.zeros((self.h, self.w, 3), np.uint8)
        base[:] = (80, 112, 53)                                   # surround green
        a, b = self.m2p((-HALF_WIDTH, -HALF_LENGTH)), self.m2p((HALF_WIDTH, HALF_LENGTH))
        cv2.rectangle(base, a, b, (152, 92, 42), -1)              # court blue
        for p, q in COURT_LINES_M:
            cv2.line(base, self.m2p(p), self.m2p(q), (245, 245, 245), 1, cv2.LINE_AA)
        cv2.line(base, self.m2p(NET_LINE_M[0]), self.m2p(NET_LINE_M[1]), (30, 30, 30), 2, cv2.LINE_AA)
        self.base = base

    def m2p(self, xy):
        return (int(round(self.w / 2 + xy[0] * self.scale)), int(round(self.h / 2 + xy[1] * self.scale)))

    def render(self, players: dict, trails: dict, ball, bounce_marks) -> np.ndarray:
        img = self.base.copy()
        r = max(4, self.w // 32)
        for side, tr in trails.items():
            pts = [self.m2p(p) for p in tr]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], COL[side], 1, cv2.LINE_AA)
        for side, xy in players.items():
            if xy is not None:
                c = self.m2p(xy)
                cv2.circle(img, c, r, COL[side], -1, cv2.LINE_AA)
                cv2.circle(img, c, r, (255, 255, 255), 1, cv2.LINE_AA)
        for xy in bounce_marks:
            cv2.circle(img, self.m2p(xy), max(3, r - 2), COL["ball"], 1, cv2.LINE_AA)
        if ball is not None:
            cv2.circle(img, self.m2p(ball), max(3, r - 2), COL["ball"], -1, cv2.LINE_AA)
        return img


class Annotator:
    def __init__(self, frame_w: int, frame_h: int, cfg: dict | None = None):
        cfg = cfg or {}
        self.show_kp = cfg.get("keypoints", True)
        self.show_boxes = cfg.get("boxes", True)
        self.show_all = cfg.get("other_people", False)
        self.trail_len = int(cfg.get("ball_trail", 15))
        self.show_mini = cfg.get("mini_court", True)
        self.show_stats = cfg.get("stats", True)
        self.stats_pos = cfg.get("stats_position", "bottom-right")
        self.W, self.H = frame_w, frame_h
        self.s = frame_w / 1280.0                                   # UI scale factor
        self.mini = MiniCourt(int(cfg.get("mini_court_width", 200) * self.s))
        self.ball_trail: deque = deque(maxlen=self.trail_len)
        self.court_trails = {NEAR: deque(maxlen=60), FAR: deque(maxlen=60)}
        self.hit_flash: tuple | None = None
        self.proc_fps = 0.0

    def draw(self, rec, pipe) -> np.ndarray:
        img = rec.frame.copy()
        s = self.s
        if rec.cut:
            self.ball_trail.clear()
            for d in self.court_trails.values():
                d.clear()

        # court keypoints
        if self.show_kp and rec.keypoints is not None and rec.H is not None:
            for i, (x, y) in enumerate(rec.keypoints):
                c = (int(x), int(y))
                cv2.circle(img, c, int(5 * s), COL["kp"], max(1, int(2 * s)), cv2.LINE_AA)
                _text(img, str(i), (c[0] + 7 * s, c[1] - 6 * s), 0.4 * s, COL["kp"], 1, bg=(15, 12, 10), pad=2)

        # other people (umpire, ball kids...) in grey
        if self.show_all:
            chosen = {id(d) for d in rec.players.values() if d is not None}
            for d in rec.all_players:
                if id(d) not in chosen:
                    x1, y1, x2, y2 = d.box.astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 1)

        # players
        if self.show_boxes:
            for side, d in rec.players.items():
                if d is None:
                    continue
                x1, y1, x2, y2 = d.box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), COL[side], max(2, int(2 * s)))
                cv2.circle(img, tuple(d.foot.astype(int)), int(4 * s), COL[side], -1)
                name = "P1" if side == NEAR else "P2"
                tid = f" #{d.track_id}" if d.track_id is not None else ""
                label = f"{name}{tid} {d.conf:.2f} | {rec.speed_kmh.get(side, 0):.1f} km/h"
                _text(img, label, (x1 + 3, y1 - 6 * s), 0.5 * s, (10, 10, 10), max(1, int(s)), bg=COL[side])

        # ball + trail
        if rec.ball is not None:
            self.ball_trail.append(rec.ball.center.copy())
        elif self.ball_trail:
            self.ball_trail.popleft()
        pts = list(self.ball_trail)
        for i in range(1, len(pts)):
            a = i / len(pts)
            col = tuple(int(c * a) for c in COL["ball"])
            cv2.line(img, tuple(pts[i - 1].astype(int)), tuple(pts[i].astype(int)), col, max(1, int(2 * s)), cv2.LINE_AA)
        if rec.ball is not None:
            x1, y1, x2, y2 = rec.ball.box.astype(int)
            pad = int(4 * s)
            if rec.ball.interpolated:
                for k in range(x1 - pad, x2 + pad, 6):
                    cv2.line(img, (k, y1 - pad), (min(k + 3, x2 + pad), y1 - pad), COL["ball"], 1)
                    cv2.line(img, (k, y2 + pad), (min(k + 3, x2 + pad), y2 + pad), COL["ball"], 1)
                lbl = "ball interp"
            else:
                cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), COL["ball"], max(1, int(2 * s)))
                lbl = f"ball {rec.ball.conf:.2f}"
            _text(img, lbl, (x2 + 8 * s, y1), 0.45 * s, (10, 10, 10), 1, bg=COL["ball"], pad=3)

        # hit flash
        for h in rec.hits:
            self.hit_flash = (h, rec.idx)
        if self.hit_flash and rec.idx - self.hit_flash[1] < int(pipe.fps * 0.9):
            h = self.hit_flash[0]
            msg = f"HIT {'P1' if h.side == NEAR else 'P2'}"
            if h.closed is not None and h.closed.speed_kmh is not None:
                msg += f"  |  last shot {h.closed.speed_kmh:.0f} km/h"
            _text(img, msg, (self.W / 2 - 40 * s, 40 * s), 0.8 * s,
                  (10, 10, 10), 2, bg=COL[h.side], pad=6)

        # mini court
        if self.show_mini:
            for side in (NEAR, FAR):
                xy = rec.player_court.get(side)
                if xy is not None:
                    self.court_trails[side].append(xy.copy())
            ball_xy = None if rec.ball is None or rec.ball.court is None else rec.ball.court
            if ball_xy is not None and (abs(ball_xy[0]) > 12 or abs(ball_xy[1]) > 20):
                ball_xy = None
            hits = [h.player_court for h in pipe.shots.hits[-6:] if h.player_court is not None]
            mini = self.mini.render(rec.player_court, self.court_trails, ball_xy, [])
            for i in range(1, len(hits)):
                cv2.line(mini, self.mini.m2p(hits[i - 1]), self.mini.m2p(hits[i]), COL["ball"], 1, cv2.LINE_AA)
            m = int(16 * s)
            x0, y0 = self.W - self.mini.w - m, m
            if y0 + self.mini.h <= self.H:
                _blend_rect(img, (x0 - 4, y0 - 4), (x0 + self.mini.w + 4, y0 + self.mini.h + 4), (0, 0, 0), 0.5)
                img[y0:y0 + self.mini.h, x0:x0 + self.mini.w] = mini

        # stats panel
        if self.show_stats:
            self._stats(img, rec, pipe)
        return img

    def _stats(self, img, rec, pipe):
        s = self.s
        rows = [("", "P1 near", "P2 far")]
        st = {side: pipe.player_stats(side) for side in (NEAR, FAR)}
        f = lambda v, u="": "-" if v is None else f"{v:.1f}{u}"
        rows += [
            ("Speed km/h", f(st[NEAR]["speed_kmh"]), f(st[FAR]["speed_kmh"])),
            ("Avg moving", f(st[NEAR]["avg_speed_moving_kmh"]), f(st[FAR]["avg_speed_moving_kmh"])),
            ("Top speed", f(st[NEAR]["top_speed_kmh"]), f(st[FAR]["top_speed_kmh"])),
            ("Distance m", f(st[NEAR]["distance_m"]), f(st[FAR]["distance_m"])),
            ("Shots", str(st[NEAR]["shots"]), str(st[FAR]["shots"])),
            ("Last shot", f(st[NEAR]["last_shot_speed_kmh"]), f(st[FAR]["last_shot_speed_kmh"])),
            ("Avg shot", f(st[NEAR]["avg_shot_speed_kmh"]), f(st[FAR]["avg_shot_speed_kmh"])),
        ]
        lh = int(22 * s)
        cw = [int(110 * s), int(78 * s), int(78 * s)]
        pw, ph = sum(cw) + int(20 * s), lh * len(rows) + int(30 * s)
        m = int(16 * s)
        x0 = m if "left" in self.stats_pos else self.W - pw - m
        y0 = m if "top" in self.stats_pos else self.H - ph - m
        _blend_rect(img, (x0, y0), (x0 + pw, y0 + ph), COL["panel"], 0.72)
        for r, row in enumerate(rows):
            y = y0 + int(22 * s) + r * lh
            x = x0 + int(10 * s)
            for c, cell in enumerate(row):
                col = COL["text"] if r else (COL[NEAR] if c == 1 else COL[FAR] if c == 2 else COL["text"])
                _text(img, cell, (x, y), 0.48 * s, col, 1)
                x += cw[c]
        res = ("court: lost" if rec.H is None else "court: fixed (clicked)" if rec.court_residual_cm < 0.05
               else f"court fit {rec.court_residual_cm:.1f} cm")
        info = f"{res} | {self.proc_fps:.1f} fps | f{rec.idx}"
        _text(img, info, (x0 + int(10 * s), y0 + ph - int(8 * s)), 0.4 * s, (170, 180, 190), 1)

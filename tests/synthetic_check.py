"""End-to-end check of the pipeline WITHOUT trained weights.

Renders a synthetic broadcast-style rally with known ground truth, feeds the
pipeline with stand-in "models" that return what YOLO26 / ResNet50 would
(noisy boxes, noisy keypoints, missed and false ball detections, an umpire),
and compares the pipeline's court positions, speeds and hits with the truth.

  python tests/synthetic_check.py                # writes runs/synthetic_annotated.mp4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tennis.analysis import FAR, NEAR                                    # noqa: E402
from tennis.court import COURT_KEYPOINTS_M, COURT_LINES_M, HALF_LENGTH   # noqa: E402
from tennis.detectors import Detection                                   # noqa: E402
from tennis.drawing import Annotator                                     # noqa: E402
from tennis.pipeline import TennisPipeline                               # noqa: E402

W, H, FPS = 1280, 720, 30
rng = np.random.default_rng(7)

# ------------------------------------------------------------------ camera
CAM, TGT, FOC, CX, CY = np.array([0, 30, 14.]), np.array([0, -2.2, 0.]), 1220, 640, 300
fw = (TGT - CAM) / np.linalg.norm(TGT - CAM)
rt = np.cross([0, 0, 1], fw); rt /= np.linalg.norm(rt)
up = np.cross(fw, rt)


def P(x, y, z=0.0):
    d = np.array([x, y, z]) - CAM
    zc = d @ fw
    return np.array([CX + FOC * (d @ rt) / zc, CY - FOC * (d @ up) / zc])


# ------------------------------------------------------------------ ground-truth rally
def plan(duration=20.0):
    shots, t, hitter = [], 1.0, NEAR
    P0 = np.array([1.0, 12.2, 2.7])
    while t < duration - 2:
        hs = 1 if hitter == NEAR else -1
        serve = not shots
        if serve:
            B = np.array([-rng.uniform(1, 3), -hs * rng.uniform(4.8, 6)]); v = rng.uniform(42, 50); h = .3
        else:
            B = np.array([rng.uniform(-3.5, 3.5), -hs * rng.uniform(7.5, 11)]); v = rng.uniform(24, 34); h = rng.uniform(1.2, 2)
        t1 = np.linalg.norm([B[0] - P0[0], B[1] - P0[1], P0[2]]) / v
        d = (B - P0[:2]) / np.linalg.norm(B - P0[:2])
        yc = -hs * rng.uniform(11.8, 13)
        C = B + d * (yc - B[1]) / d[1]; C[0] = np.clip(C[0], -5.2, 5.2)
        t2 = max(np.linalg.norm(C - B) / (v * .6), 1.1 - t1)
        shots.append(dict(hitter=hitter, t=t, P=P0.copy(), B=B, tb=t + t1, C=np.r_[C, 1.0], tn=t + t1 + t2, h=h,
                          h2=rng.uniform(.7, 1.3), v=v))
        t, P0, hitter = t + t1 + t2, np.r_[C, 1.0], 1 - hitter
    return shots


shots = plan()
DUR = shots[-1]["tn"]
N = int(DUR * FPS)


def ball_at(t):
    s0 = shots[0]
    if t < s0["t"]:
        w = max(0.0, (t - (s0["t"] - 1)))
        return np.array([s0["P"][0], s0["P"][1], 1.2 + 1.5 * w + 4 * w * (1 - w)])
    for s in shots:
        if s["t"] <= t < s["tn"]:
            if t < s["tb"]:
                u = (t - s["t"]) / (s["tb"] - s["t"])
                g = s["P"][:2] + (s["B"] - s["P"][:2]) * u
                return np.r_[g, (1 - u) * s["P"][2] + 4 * s["h"] * u * (1 - u)]
            u = (t - s["tb"]) / (s["tn"] - s["tb"])
            g = s["B"] + (s["C"][:2] - s["B"]) * u
            return np.r_[g, u * s["C"][2] + 4 * s["h2"] * u * (1 - u)]
    return None


# players move toward the next contact point with capped speed/acceleration
pos = {NEAR: np.array([0.7, 12.5]), FAR: np.array([-2.8, -12.9])}
vel = {NEAR: np.zeros(2), FAR: np.zeros(2)}
gt_pos, gt_ball = [], []
for f in range(N):
    t = f / FPS
    act = max([s for s in shots if s["t"] <= t], key=lambda s: s["t"], default=None)
    for side in (NEAR, FAR):
        sign = 1 if side == NEAR else -1
        tgt, vmax = pos[side], 0.0
        if act is not None:
            if act["hitter"] == side:
                tgt, vmax = np.array([act["P"][0] * .3, sign * 12.7]), 4.5
            else:
                tgt = act["C"][:2] + np.array([0.7 * sign, 0.2 * sign])
                vmax = min(7.5, np.linalg.norm(tgt - pos[side]) / max(act["tn"] - t, .12))
        d = tgt - pos[side]; dist = np.linalg.norm(d)
        want = d / dist * min(vmax, dist * 3) if dist > 1e-3 else np.zeros(2)
        dv = want - vel[side]; n = np.linalg.norm(dv); amax = 16 / FPS
        vel[side] += dv * (min(1, amax / n) if n > 0 else 0)
        pos[side] = pos[side] + vel[side] / FPS
    gt_pos.append({k: v.copy() for k, v in pos.items()})
    gt_ball.append(ball_at(t))


# ------------------------------------------------------------------ renderer
def render(f):
    img = np.zeros((H, W, 3), np.uint8)
    wall = int(P(0, -18)[1])
    img[:wall] = (40, 30, 20)
    img[wall:] = (80, 120, 55)
    c = np.array([P(x, y) for x, y in [(-5.485, -11.885), (5.485, -11.885), (5.485, 11.885), (-5.485, 11.885)]], np.int32)
    cv2.fillConvexPoly(img, c, (150, 90, 40))
    for p, q in COURT_LINES_M:
        cv2.line(img, tuple(P(*p).astype(int)), tuple(P(*q).astype(int)), (240, 240, 240), 2, cv2.LINE_AA)
    items = []
    for side in (NEAR, FAR):
        x, y = gt_pos[f][side]
        items.append((y, "p", side, x))
    b = gt_ball[f]
    if b is not None:
        items.append((b[1], "b", b, 0))
    items.append((0.0, "n", None, 0))
    for y, kind, a, x in sorted(items, key=lambda i: i[0]):
        if kind == "p":
            col = (235, 235, 235) if a == NEAR else (50, 60, 210)
            foot, head = P(x, y), P(x, y, 1.85)
            wpx = int(abs(P(x + .3, y)[0] - P(x - .3, y)[0]))
            cv2.rectangle(img, (int(foot[0]) - wpx // 2, int(head[1])), (int(foot[0]) + wpx // 2, int(foot[1])), col, -1)
            cv2.circle(img, tuple(P(x, y, 1.75).astype(int)), max(3, wpx // 3), (120, 160, 200), -1)
        elif kind == "b":
            p = P(*a)
            cv2.circle(img, tuple(p.astype(int)), 4, (60, 240, 230), -1, cv2.LINE_AA)
        else:
            l, r, cl, cr = P(-6.4, 0), P(6.4, 0), P(-6.4, 0, 1.07), P(6.4, 0, 1.07)
            cv2.fillConvexPoly(img, np.array([l, r, cr, cl], np.int32), (40, 40, 40))
            cv2.line(img, tuple(cl.astype(int)), tuple(cr.astype(int)), (245, 245, 245), 2)
    return img


# ------------------------------------------------------------------ stand-in models
class Clock:
    f = 0


class StubPlayers:
    """Behaves like YOLO26 + ByteTrack: two players with stable IDs plus an umpire at the net post."""
    def __call__(self, frame):
        out = []
        for side, tid in ((NEAR, 1), (FAR, 2)):
            x, y = gt_pos[Clock.f][side]
            pts = np.array([P(x + dx, y, z) for dx in (-.38, .38) for z in (0, 1.95)])
            box = np.r_[pts.min(0), pts.max(0)] + rng.normal(0, 1.4, 4)
            out.append(Detection(box, float(rng.uniform(.86, .97)), 0, tid))
        u = np.array([P(dx, 0.5, z) for dx in (7.6, 8.2) for z in (0, 1.8)])
        out.append(Detection(np.r_[u.min(0), u.max(0)], 0.8, 0, 3))
        return out

    def reset(self):
        pass


class StubBall:
    """Behaves like a YOLO26 ball model: 8% misses and occasional false positives."""
    def __call__(self, frame, prefer_near=None, max_dist_px=None):
        b = gt_ball[Clock.f]
        if b is None or rng.random() < 0.08:
            return None
        if rng.random() < 0.01:
            c = rng.uniform([0, 0], [W, H])
        else:
            c = P(*b) + rng.normal(0, .8, 2)
        return Detection(np.r_[c - 5, c + 5], float(rng.uniform(.4, .9)))


KP_TRUE = np.array([P(x, y) for x, y in COURT_KEYPOINTS_M], np.float32)


class StubKeypoints:
    """Behaves like the ResNet50 regressor: 14 points with ~1.5 px noise."""
    def __call__(self, frame):
        return KP_TRUE + rng.normal(0, 1.5, KP_TRUE.shape).astype(np.float32)


# ------------------------------------------------------------------ run
def main():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    out_dir = ROOT / "runs"; out_dir.mkdir(exist_ok=True)
    annot = Annotator(W, H, cfg.get("draw", {}))
    pipe = TennisPipeline(StubPlayers(), StubBall(), StubKeypoints(), cfg, FPS, annot)
    vw = cv2.VideoWriter(str(out_dir / "synthetic_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    recs = []
    for f in range(N):
        Clock.f = f
        for r in pipe.process(render(f), f / FPS):
            vw.write(r.annotated); recs.append(r)
    for r in pipe.flush():
        vw.write(r.annotated); recs.append(r)
    vw.release()
    snap = recs[int(N * 0.55)].annotated
    cv2.imwrite(str(out_dir / "synthetic_frame.jpg"), snap)

    # ---- compare with ground truth
    errs = [np.linalg.norm(r.player_court[s] - gt_pos[r.idx][s]) for r in recs for s in (NEAR, FAR)
            if r.player_court.get(s) is not None]
    gt_speed = {s: [np.linalg.norm(gt_pos[f][s] - gt_pos[f - 5][s]) / (5 / FPS) * 3.6 for f in range(5, N)] for s in (NEAR, FAR)}
    sp_err = [abs(recs[f].speed_kmh[s] - gt_speed[s][f - 5]) for f in range(30, N) for s in (NEAR, FAR)]
    gt_dist = {s: sum(np.linalg.norm(gt_pos[f][s] - gt_pos[f - 1][s]) for f in range(1, N)) for s in (NEAR, FAR)}
    filled = sum(1 for r in recs if r.ball is not None and r.ball.interpolated)
    hits = pipe.shots.hits
    gt_shot_speeds = [np.linalg.norm(shots[i + 1]["P"][:2] - shots[i]["P"][:2]) / (shots[i + 1]["t"] - shots[i]["t"]) * 3.6
                      for i in range(len(shots) - 1)]
    report = {
        "frames": N,
        "court_fit_cm_median": round(float(np.nanmedian([r.court_residual_cm for r in recs])), 2),
        "player_position_error_m": {"median": round(float(np.median(errs)), 3), "p95": round(float(np.percentile(errs, 95)), 3)},
        "player_speed_error_kmh_median": round(float(np.median(sp_err)), 2),
        "distance_m": {"P1": [round(pipe.motion[NEAR].distance_m, 1), "truth", round(gt_dist[NEAR], 1)],
                       "P2": [round(pipe.motion[FAR].distance_m, 1), "truth", round(gt_dist[FAR], 1)]},
        "ball_frames_interpolated": filled,
        "hits_detected": len(hits), "hits_truth": len(shots),
        "hit_sides_correct": sum(1 for h in hits for s in shots
                                 if abs(h.t - s["t"]) < 0.25 and h.side == s["hitter"]),
        "hits_missed_at_s": [round(s["t"], 2) for s in shots if not any(abs(h.t - s["t"]) < 0.25 for h in hits)],
        "shot_speed_kmh": {"detected": [None if h.speed_kmh is None else round(h.speed_kmh) for h in hits],
                           "truth": [round(v) for v in gt_shot_speeds]},
        "umpire_ignored": all(r.players[s] is None or r.players[s].track_id in (1, 2) for r in recs for s in (NEAR, FAR)),
    }
    print(json.dumps(report, indent=2))
    (out_dir / "synthetic_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

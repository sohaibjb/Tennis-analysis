"""Tennis match analysis: players + ball tracking, court mapping, speeds and shots.

Examples
  python main.py --source input_videos/match.mp4
  python main.py --source input_videos/match.mp4 --show --max-frames 600
  python main.py --source 0 --show                       # webcam
  python main.py --source rtsp://user:pass@cam/stream --show --realtime
  python main.py --source match.mp4 --court runs/match_court.json   # no keypoint model: clicked corners

Missing player / ball weights fall back to the public COCO YOLO26 model (person / sports ball).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import yaml

from tennis.analysis import FAR, NEAR
from tennis.detectors import BallDetector, CourtKeypointModel, ManualCourt, PlayerDetector, resolve_device
from tennis.drawing import Annotator
from tennis.pipeline import TennisPipeline
from tennis.video_io import VideoSink, VideoSource


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, webcam index (0) or stream URL")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output", default=None, help="annotated video path (default: runs/<name>_annotated.mp4)")
    ap.add_argument("--no-save", action="store_true", help="don't write the annotated video")
    ap.add_argument("--show", action="store_true", help="display the annotated stream in a window")
    ap.add_argument("--realtime", action="store_true", help="live sources: always process the newest frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="also write per-frame tracks as JSON lines")
    ap.add_argument("--device", default=None, help="cuda:0, cpu, mps (overrides config)")
    ap.add_argument("--court", default=None,
                    help="court corners JSON from calibrate_court.py (fixed camera; replaces the keypoint model)")
    return ap.parse_args()


def _weights(section: dict, name: str) -> tuple[str, list | None]:
    """Use the trained weights if present, otherwise the public COCO fallback (with a warning)."""
    w = section["weights"]
    if Path(w).exists():
        return w, section.get("classes")
    fb = section.get("fallback_weights")
    if not fb:
        raise SystemExit(f"[setup] {name} weights not found: {w}")
    print(f"[setup] {name} weights not found at {w} -> using public {fb} classes {section.get('fallback_classes')}"
          f" (works, but your trained model will be more accurate)")
    return fb, section.get("fallback_classes")


def build_models(cfg, device, court_json=None):
    m = cfg["models"]
    half = bool(cfg.get("half", False)) and device.startswith("cuda")
    p, b, k = m["players"], m["ball"], m["keypoints"]
    pw, pc = _weights(p, "player")
    players = PlayerDetector(pw, device, p.get("conf", 0.35), p.get("imgsz", 640),
                             pc, p.get("tracker", "bytetrack.yaml"), half)
    bw, bc = _weights(b, "ball")
    ball = BallDetector(bw, device, b.get("conf", 0.15), b.get("imgsz", 1280), bc, half)
    if court_json:
        print(f"[setup] court from clicked corners: {court_json}")
        kps = ManualCourt.from_json(court_json)
    elif Path(k["weights"]).exists():
        kps = CourtKeypointModel(k["weights"], device, k.get("input_size", 224), k.get("num_keypoints", 14),
                                 k.get("output", "pixels"), k.get("imagenet_normalize", True))
    else:
        raise SystemExit(
            f"[setup] court keypoint weights not found: {k['weights']}\n"
            "  Either put your ResNet50 keypoint model there (see weights/README.txt),\n"
            "  or, for a fixed camera, click the 4 court corners once:\n"
            "    python calibrate_court.py --source <video>\n"
            "    python main.py --source <video> --court runs/<video>_court.json")
    return players, ball, kps


def frame_json(rec):
    def box(d):
        return None if d is None else {"box": [round(float(v), 1) for v in d.box], "conf": round(d.conf, 3),
                                       "track_id": d.track_id}
    xy = lambda v: None if v is None else [round(float(a), 3) for a in v]
    return {
        "frame": rec.idx, "t": round(rec.t, 4), "cut": rec.cut,
        "court_fit_cm": None if rec.H is None else round(rec.court_residual_cm, 2),
        "keypoints_px": None if rec.keypoints is None else [[round(float(a), 1) for a in p] for p in rec.keypoints],
        "players": {name: {**(box(rec.players.get(side)) or {}), "court_xy_m": xy(rec.player_court.get(side)),
                           "speed_kmh": round(rec.speed_kmh.get(side, 0.0), 2)}
                    for side, name in ((NEAR, "P1"), (FAR, "P2"))},
        "ball": None if rec.ball is None else {
            "center_px": xy(rec.ball.center), "conf": round(rec.ball.conf, 3),
            "interpolated": rec.ball.interpolated, "court_xy_m_ground": xy(rec.ball.court)},
        "timings_ms": {k: round(v, 2) for k, v in rec.timings_ms.items()},
    }


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    print(f"[setup] device={device}")

    src = VideoSource(args.source, realtime=args.realtime)
    print(f"[setup] source {args.source}: {src.width}x{src.height} @ {src.fps:.1f} fps"
          + (f", {src.total} frames" if src.total else " (live)"))

    players, ball, kps = build_models(cfg, device, args.court)
    annot = Annotator(src.width, src.height, cfg.get("draw", {}))
    pipe = TennisPipeline(players, ball, kps, cfg, src.fps, annot)

    out_dir = Path(cfg.get("output_dir", "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "live" if src.live else Path(str(args.source)).stem
    sink = None
    if not args.no_save:
        out_path = args.output or str(out_dir / f"{stem}_annotated.mp4")
        sink = VideoSink(out_path, src.fps, (src.width, src.height))
        print(f"[setup] writing {out_path}")
    jf = open(out_dir / f"{stem}_tracks.jsonl", "w") if args.json else None

    def emit(recs):
        for r in recs:
            if sink:
                sink.write(r.annotated)
            if jf:
                jf.write(json.dumps(frame_json(r)) + "\n")
            if args.show:
                cv2.imshow("Tennis analysis", r.annotated)
            for h in r.hits:
                msg = f"[hit] {h.t:7.2f}s  {'P1' if h.side == NEAR else 'P2'}"
                if h.closed is not None and h.closed.speed_kmh is not None:
                    msg += f"   (previous shot by {'P1' if h.closed.side == NEAR else 'P2'}: {h.closed.speed_kmh:.0f} km/h)"
                print(("\n" if src.total else "") + msg)
        if args.show and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            raise KeyboardInterrupt

    t_last, n_last = time.perf_counter(), 0
    try:
        for i, (frame, ts) in enumerate(src):
            if args.max_frames and i >= args.max_frames:
                break
            emit(pipe.process(frame, ts))
            if i - n_last >= 15:
                now = time.perf_counter()
                annot.proc_fps = (i - n_last) / (now - t_last)
                t_last, n_last = now, i
                if src.total:
                    print(f"\r[run] {i}/{src.total} frames  {annot.proc_fps:.1f} fps", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[run] stopped by user")
    emit(pipe.flush())
    print()

    src.release()
    if sink:
        sink.release()
    if jf:
        jf.close()
    if args.show:
        cv2.destroyAllWindows()

    summary = pipe.summary()
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / f"{stem}_shots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "player", "speed_kmh", "court_x_m", "court_y_m"])
        for s in summary["shots"]:
            xy = s["player_court_xy_m"] or [None, None]
            w.writerow([s["frame"], s["time_s"], s["player"], s["speed_kmh"], *xy])
    for name, st in summary["players"].items():
        print(f"{name}: dist {st['distance_m']} m, top {st['top_speed_kmh']} km/h, "
              f"avg {st['avg_speed_moving_kmh']} km/h, shots {st['shots']}, avg shot {st['avg_shot_speed_kmh']} km/h")
    print(f"[done] outputs in {out_dir}/")


if __name__ == "__main__":
    main()

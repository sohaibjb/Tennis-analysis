"""Click the 4 court corners once, so a fixed-camera video can be analysed without a keypoint model.

  python calibrate_court.py --source input_videos/match.mp4
  python calibrate_court.py --source input_videos/match.mp4 --frame 300     # pick a frame with a clear court

Click the outer (doubles) corners in this order:
  1  far-left     2  far-right     3  near-left     4  near-right
The full court model is drawn over the frame after the 4th click so you can check it.
Keys:  Enter / s = save    r = start again    q / Esc = quit without saving

Then run:  python main.py --source input_videos/match.mp4 --court runs/match_court.json

On a machine without a display, read the corner pixels from a screenshot and pass them directly:
  python calibrate_court.py --source match.mp4 --corners 575,305 1340,305 285,860 1630,860
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tennis.court import COURT_KEYPOINTS_M, COURT_LINES_M, NET_LINE_M

NAMES = ["far-left", "far-right", "near-left", "near-right"]


def court_overlay(img, corners, scale=1.0):
    H = cv2.getPerspectiveTransform(COURT_KEYPOINTS_M[:4].copy(), np.float32(corners))
    proj = lambda pts: cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2) * scale
    for p, q in COURT_LINES_M + [NET_LINE_M]:
        a, b = proj([p, q])
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(proj(COURT_KEYPOINTS_M)):
        cv2.circle(img, (int(x), int(y)), 5, (32, 176, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 176, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=0, help="frame number to calibrate on")
    ap.add_argument("--output", default=None, help="default: runs/<video>_court.json")
    ap.add_argument("--corners", nargs=4, default=None, metavar="X,Y",
                    help="skip clicking: far-left far-right near-left near-right")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, str) and not src.startswith(("rtsp", "http")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.source}")

    stem = "live" if isinstance(src, int) else Path(str(args.source)).stem
    out = Path(args.output or f"runs/{stem}_court.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.corners:
        corners = [list(map(float, c.split(","))) for c in args.corners]
    else:
        h, w = frame.shape[:2]
        scale = min(1.0, 1400 / w, 850 / h)                      # fit large frames on screen
        disp_base = cv2.resize(frame, (int(w * scale), int(h * scale)))
        corners: list[list[float]] = []

        def on_mouse(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x / scale, y / scale])

        cv2.namedWindow("calibrate", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("calibrate", on_mouse)
        while True:
            disp = disp_base.copy()
            for i, (x, y) in enumerate(corners):
                cv2.circle(disp, (int(x * scale), int(y * scale)), 6, (0, 0, 255), -1)
                cv2.putText(disp, str(i + 1), (int(x * scale) + 8, int(y * scale) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(corners) == 4:
                court_overlay(disp, corners, scale)
                msg = "Check the overlay. Enter/s = save, r = redo, q = quit"
            else:
                msg = f"Click corner {len(corners) + 1}: {NAMES[len(corners)]} (outer doubles corner)"
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(disp, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("calibrate", disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                cv2.destroyAllWindows()
                raise SystemExit("Quit without saving.")
            if k == ord("r"):
                corners.clear()
            if k in (13, 10, ord("s")) and len(corners) == 4:
                break
        cv2.destroyAllWindows()

    out.write_text(json.dumps({"source": str(args.source), "frame": args.frame,
                               "order": NAMES, "corners_px": corners}, indent=2))
    check = out.with_suffix(".jpg")
    cv2.imwrite(str(check), court_overlay(frame.copy(), corners))
    print(f"Saved {out}  (overlay preview: {check})")
    print(f"Next: python main.py --source {args.source} --court {out}")


if __name__ == "__main__":
    main()

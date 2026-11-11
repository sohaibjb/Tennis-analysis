# Tennis Match Analysis

Turns tennis footage — a recorded video **or** a live stream — into an annotated feed:

- detects and tracks both players (YOLO26 + ByteTrack), ignoring umpire, ball kids and crowd
- detects the ball (YOLO26) and fills short gaps by interpolation
- finds the 14 court keypoints (ResNet50) and fits an image → court homography every frame
- maps players and ball onto a top-down **mini court**
- measures each player's **speed, top speed, average speed and distance covered**
- detects **hits** and estimates **shot speed**
- writes an annotated video, per-frame tracks (JSON lines), a shot list (CSV) and a match summary (JSON)

## Output highlight

[![Annotated output highlight](demo/vid1_highlight.gif)](demo/vid1_annotated.mp4)

Click the preview to open the full annotated output video.

```
frame ──► YOLO26 players ──► pick 2 players (one per half) ─┐
      ├─► YOLO26 ball ─────► outlier filter + gap filling ──┤
      └─► ResNet50 keypoints ─► RANSAC homography ──────────┴─► court coordinates (metres)
                                                                 ├─► speed / distance per player
                                                                 ├─► hits + shot speed
                                                                 └─► annotated frame + mini court
```

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch first (see pytorch.org), then the requirements.
On a server without a screen, replace `opencv-python` with `opencv-python-headless` and don't use `--show`.

## 2. Models: three ways to start

**A. You have your trained weights.** Copy them into `weights/`:

| file | model | what the code expects |
|---|---|---|
| `weights/players_yolo26.pt` | Ultralytics YOLO26 | standard Ultralytics `.pt` from training (`runs/detect/train/weights/best.pt`) |
| `weights/ball_yolo26.pt` | Ultralytics YOLO26 | same |
| `weights/keypoints_resnet50.pth` | ResNet50 | torchvision `resnet50` with `fc = Linear(2048, 28)`, saved as a `state_dict` **or** a whole model |

**B. You don't have them yet: run anyway.**
- Missing player / ball weights fall back automatically to the public COCO model `yolo26s.pt`
  (downloaded on first run; classes `person` and `sports ball`). A warning is printed.
- Missing keypoint weights: for a fixed camera, click the 4 court corners once:
  ```bash
  python calibrate_court.py --source input_videos/match.mp4
  python main.py --source input_videos/match.mp4 --court runs/match_court.json
  ```
  The overlay drawn after the 4th click lets you check the fit before saving.

**C. Train your own.** `training/README.md` has the datasets and one command per model
(`training/train_yolo.py` for players and ball, `training/train_keypoints.py` for the court).

Check these settings in `config.yaml` against how your models were trained:

- **Players / ball `classes`**: `null` uses every class.
- **Ball `imgsz`**: the size you trained at (1280 helps tiny balls).
- **Keypoints `input_size`** (default 224), **`output`** (`pixels` = coordinates in the resized input image, `normalized` = 0–1), **`imagenet_normalize`**.
- **Keypoint order**: the TennisCourtDetector order drawn in `tennis/court.py`
  (0–3 doubles corners, 4–7 singles sideline ends, 8–11 service line ends, 12–13 centre line ends).
  If yours differs, set `keypoints.order` in `config.yaml`.

## 3. Run

```bash
# recorded match
python main.py --source input_videos/match.mp4

# watch while it runs, stop with q / Esc
python main.py --source input_videos/match.mp4 --show

# webcam or capture card
python main.py --source 0 --show

# live stream (RTSP / HTTP / UDP); --realtime drops frames instead of lagging behind
python main.py --source "rtsp://user:pass@camera-ip/stream" --show --realtime

# also export per-frame tracks
python main.py --source match.mp4 --json
```

Outputs in `runs/` (on Windows use backslashes, e.g. `input_videos\\match.mp4`):

| file | contents |
|---|---|
| `<name>_annotated.mp4` | video with boxes, IDs, speeds, ball trail, keypoints, mini court, stats panel |
| `<name>_summary.json` | per player: distance, top / avg speed, shots, avg / max shot speed; list of shots |
| `<name>_shots.csv` | one row per detected hit: time, player, shot speed, player position on court |
| `<name>_tracks.jsonl` | (with `--json`) every frame: player boxes + court positions + speed, ball, keypoints, timings |

## 4. Tested on

- A real ATP broadcast clip (1920×1080, 30 fps), public fallback models and clicked corners: both players
  tracked with the umpire and ball kids ignored, ball present in 99 % of frames after gap filling,
  hits attributed to the right player.
- A synthetic rally with known ground truth (below).

## 5. Test without any weights

```bash
python tests/synthetic_check.py
```

Renders a synthetic rally with known ground truth, feeds the pipeline with stand-in models that behave like
yours (noisy boxes and keypoints, missed and false ball detections, an umpire next to the net) and compares
the results with the truth. It also writes `runs/synthetic_annotated.mp4`. Typical result:

| check | result |
|---|---|
| court fit residual | ~5 cm |
| player position error | ~7 cm median, ~28 cm p95 |
| player speed error | ~1.5 km/h median |
| distance covered | within ~10 % |
| hits found / correct player | 15–16 of 16 |
| umpire ignored | yes |

## How the numbers are computed

- **Player position**: bottom-centre of the player box (the feet) mapped through the homography, then smoothed.
- **Speed**: displacement over 0.17 s (config `speed_window_s`), lightly smoothed. Jumps above 40 km/h are treated as tracking glitches.
- **Distance**: counted only once the player leaves a 25 cm radius around the last counted point, so box jitter (worse on the far side) doesn't add phantom metres.
- **Hits**: turning points of the ball's ground-plane track along the court. Travel between players dominates that track, so bounces and the serve toss don't trigger it. The hitter is the player on the side the ball is leaving, and the ball must be within reach of that player's box.
- **Shot speed**: court distance from the hitter (at contact) to the next hitter (at their contact) ÷ time between the contacts. It is an average over the whole flight (including the bounce), so it reads lower than radar speed off the racket, and the last shot of a rally (a winner or error) has no speed.
- **Camera cuts** (keypoints jump) reset trackers and speed history; frames where the court can't be fitted (close-ups, replays) are drawn without court mapping.

## Live use: latency and speed

- Output is delayed by `pipeline.lag_frames` (default 10 frames ≈ 0.33 s at 30 fps) so missed ball frames can be filled before drawing. Lower it for less delay, raise `ball_max_gap` with it.
- Three models per frame need a GPU for real time. On a fixed camera set `keypoints.every_n_frames: 15` (the court doesn't move), use `half: true` on CUDA, and lower `imgsz` if needed.
- `--realtime` always processes the newest camera frame, so the output never falls behind the live feed.

## Project layout

```
main.py                 CLI: source -> pipeline -> video / JSON / CSV
config.yaml             model paths and all tunable settings
tennis/detectors.py     YOLO26 player tracker, YOLO26 ball detector, ResNet50 keypoint model
tennis/court.py         court dimensions, 14 reference points, RANSAC homography
tennis/analysis.py      player selection, ball gap filling, speed/distance, hit + shot speed
tennis/pipeline.py      streaming pipeline with the look-back buffer
tennis/drawing.py       annotations, mini court, stats panel
tennis/video_io.py      file / webcam / stream input, mp4 output
calibrate_court.py      click 4 court corners (fixed camera, no keypoint model needed)
training/               train_yolo.py (players, ball), train_keypoints.py (court), README.md
tests/synthetic_check.py  end-to-end check with ground truth, no weights needed
```

## Known limits

- Players are labelled by side (P1 near, P2 far). After a change of ends the labels follow the side, not the person.
- The ball's mini-court dot is its ground-plane projection: exact when the ball touches the ground, offset when it's high in the air.
- Forehand/backhand and in/out calls aren't included; both need extra models or rules (pose for stroke type, bounce detection for line calls).

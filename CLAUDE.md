# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Computer vision pipeline that analyzes tennis match video: detects players, rackets, and ball (YOLOv8x), detects court keypoints (template homography), tracks the ball and filters out-of-bounds detections, and outputs an annotated video.

## Running the Pipeline

```bash
# Stage 1 — detect court + objects
.venv/bin/python detect.py -i <video> -m models/yolo26x.pt -s models/court_seg.pt -z 1920

# Stage 2 — ball tracking + spatial filtering
.venv/bin/python parse.py -i <video>.json

# Stage 3 — render annotated video
.venv/bin/python render.py -i <video> -j <video>_parsed.json -o <output>.mp4
```

- **Python env:** `.venv/bin/python` (Python 3.10 required, see README)
- **render.py** accepts both detect.py and parse.py output (COCO format, same schema)

## Dependencies

GTX 1080 Ti (sm_61) is not supported by PyTorch 2.x. Must use Python 3.10 + torch 1.13.1+cu117:

```bash
python3.10 -m venv .venv
.venv/bin/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
.venv/bin/pip install ultralytics==8.4.35 opencv-python numpy==1.26.4 pandas scipy tqdm PySide6
```

## Training & Evaluation

```bash
# Fine-tune
.venv/bin/python train_yolo.py --data datasets/xxx-yolo/data.yaml --name finetune

# Evaluate
.venv/bin/python eval_yolo.py --data datasets/xxx-yolo/data.yaml
```

- Training outputs to `runs/{task}/exp/<name>/` under the project root (absolute path, not affected by ultralytics global `runs_dir` setting)

## Architecture

### Pipeline Flow

```
detect.py
  → CourtDetector.predict(frame 0)     — 14 keypoints via template homography
  → CourtDetector.get_clearance_hull() — ground hull + 2 m volume hull
  → ObjectsDetector.run()              — YOLOv8x predict for person/racket/ball
  → save_coco()                        → <video>.json

parse.py
  → BallTracker.run()                  — physics-based trajectory linking
  → _filter_players/rackets/balls()    — spatial filtering by clearance hulls
  → save_coco() with valid/track_id    → <video>_parsed.json

render.py
  → load_detections()                  — reads either JSON
  → split valid/invalid per frame
  → draw: court lines + volume wireframe → valid boxes → invalid boxes (dark+X) → ball trajectories
  → ffmpeg H.264 encoding              → <output>.mp4
```

### Key Design Decisions

- **Unified JSON format:** detect.py and parse.py share the same COCO schema. parse.py extends each annotation with `track_id` (int|null) and `valid` (bool). render.py reads both; `valid` defaults to `true` when absent (detect.py output).
- **Single inference:** `model.predict()` (not `model.track()`) with `classes=[0, 38, 32]`. `track()` was abandoned because ByteTrack drops unconfirmed detections (box.id=None), causing ball loss.
- **Court keypoints** detected only on frame 0 — the court doesn't move.
- **Clearance volume:** ITF standard / 2 — back 3.20 m, side 1.83 m, height 2.0 m. Projected as wireframe in render/browse. Players filtered by ground hull (bottom-center inside), rackets by volume hull (5-point bbox check), balls by trajectory start point and static threshold (avg displacement < 5 px).
- **Invalid detections** are kept in the JSON (`valid=false`) and shown in dark color + X in render/browse.
- **Output encoding:** ffmpeg libx264, crf=18, preset=fast.
- **Inference device:** auto-selects cuda > mps > cpu.

### Module Responsibilities

| Module | Purpose |
|---|---|
| `detect.py` | Stage 1: court + object detection → COCO JSON |
| `parse.py` | Stage 2: ball tracking + spatial filtering → COCO JSON with valid/track_id |
| `render.py` | Stage 3: draw annotations on video frames → H.264 video |
| `browse.py` | Interactive frame-by-frame viewer (PySide6) for any stage JSON |
| `build_coco.py` | Extract video frames as JPEG + migrate valid annotations to training COCO JSON |
| `court_detector.py` | CourtDetector: template homography → 14 keypoints, clearance hulls; `compute_H_from_kps` |
| `objects_detector.py` | ObjectsDetector: single predict() call for person/racket/ball |
| `ball_tracker.py` | BallTracker: physics-based trajectory linking across frames |
| `utils.py` | Video I/O, `save_coco` / `load_detections`, `text_params` |
| `train_yolo.py` | Fine-tune YOLO on tennis dataset |
| `eval_yolo.py` | Evaluate model with per-class AP metrics |
| `debug_court.py` | Save intermediate court detection images for debugging |

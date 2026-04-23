# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Computer vision pipeline that analyzes tennis match video: detects players, rackets, and ball (YOLOv8x), detects court keypoints (template homography), tracks the ball and filters out-of-bounds detections, and outputs an annotated video.

## Running the Pipeline

```bash
# Stage 1 — detect court + objects
.venv/bin/python detect.py -i <video> -m models/yolo26x.pt -s models/court_seg.pt -z 1920

# Stage 2 — ball tracking (SORT-style Kalman + Hungarian)
.venv/bin/python track.py -i <video>_detected.json

# Stage 3 — spatial filtering
.venv/bin/python parse.py -i <video>_tracked.json

# Stage 4 — render annotated video
.venv/bin/python render.py -i <video> -j <video>_parsed.json -o <output>.mp4
```

- **Python env:** `.venv/bin/python` (Python 3.10 required, see README)
- **render.py** accepts any stage JSON (COCO format, same schema)

## Dependencies

GTX 1080 Ti (sm_61) is not supported by PyTorch 2.x. Must use Python 3.10 + torch 1.13.1+cu117:

```bash
python3.10 -m venv .venv
.venv/bin/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
.venv/bin/pip install ultralytics==8.4.35 opencv-python numpy==1.26.4 pandas scipy tqdm PySide6
```

## Training & Evaluation

```bash
# Fine-tune ball detector
.venv/bin/python train_ball.py --data datasets/xxx-yolo/data.yaml --name finetune

# Fine-tune court detector
.venv/bin/python train_court.py --data datasets/xxx-court/data.yaml --name finetune

# Evaluate ball detector
.venv/bin/python eval_ball.py --data datasets/xxx-yolo/data.yaml
```

- Training outputs to `runs/{task}/exp/<name>/` under the project root (absolute path, not affected by ultralytics global `runs_dir` setting)

## Architecture

### Pipeline Flow

```
detect.py
  → CourtDetector.predict(frame 0)     — 14 keypoints via template homography
  → CourtDetector.get_clearance_hull() — ground hull + 2 m volume hull
  → ObjectsDetector.run()              — YOLOv8x predict for person/racket/ball
  → save_coco()                        → <video>_detected.json

track.py
  → BallTracker.run()                  — SORT-style: Kalman filter + Hungarian matching
                                          min_hits / max_age / max_dist from physics
  → save_coco() with track_id          → <video>_tracked.json

parse.py
  → _filter_players/rackets/balls()    — spatial filtering by clearance hulls
  → save_coco() (only valid dets)      → <video>_parsed.json

render.py
  → load_detections()                  — reads any stage JSON
  → split valid/invalid per frame      — (valid defaults to true when absent)
  → draw: court lines + volume wireframe + outside wall masks → valid boxes → invalid boxes (dark+X) → ball trajectories
  → ffmpeg H.264 encoding              → <output>.mp4
```

### Key Design Decisions

- **Unified JSON format:** all stages share the same COCO schema. track.py adds `track_id` (int|null); parse.py adds `valid` (bool). render.py reads any stage; `track_id` defaults to null and `valid` defaults to true when absent.
- **Single inference:** `model.predict()` (not `model.track()`) with `classes=[0, 38, 32]`. `track()` was abandoned because ByteTrack drops unconfirmed detections (box.id=None), causing ball loss.
- **Court keypoints** detected only on frame 0 — the court doesn't move.
- **Clearance volume:** ITF standard / 2 — back 3.20 m, side 1.83 m, height 2.0 m. Projected as wireframe in render/check_json. Players filtered by ground hull (bottom-center inside), rackets by volume hull (5-point bbox check), balls by side-wall quads + static threshold (avg displacement < 5 px).
- **Side wall filtering (balls):** Moving ball trajectories are invalidated if the start point falls inside the left or right wall quadrilateral — each wall is the volume side face (`vol_bottom_pts` + `vol_top_pts`) extended to sky via the vertical edge direction. Using the wall face quad (not the full volume hull) avoids false negatives for airborne balls, and extending to sky avoids false negatives from the 2 m height cap. Static balls still use volume hull per-frame.
- **Outside wall masks:** render.py and check_json.py draw a semi-transparent dark overlay outside the two side walls. The mask boundary follows the far-left/far-right vertical edge of the volume (bottom → 2 m top → sky), creating a wall-like appearance rather than a flat curtain.
- **parse.py output** contains only valid detections (invalid ones are discarded). render.py handles both: if a stage JSON includes `valid=false` annotations, they are rendered dark+X; when `valid` is absent it defaults to true.
- **Output encoding:** ffmpeg libx264, crf=18, preset=fast.
- **Inference device:** auto-selects cuda > mps > cpu.

### Module Responsibilities

| Module | Purpose |
|---|---|
| `detect.py` | Stage 1: court + object detection → COCO JSON |
| `track.py` | Stage 2: ball tracking (SORT-style) → COCO JSON with track_id |
| `parse.py` | Stage 3: spatial filtering → COCO JSON with valid |
| `render.py` | Stage 4: draw annotations on video frames → H.264 video |
| `pose.py` | Stage 5 (optional): run pose estimation on players → COCO JSON with keypoints |
| `check_json.py` | Interactive frame-by-frame viewer (PySide6) for any stage JSON |
| `extract_dataset.py` | Extract video frames as JPEG + migrate valid annotations to training COCO JSON |
| `court_detector.py` | CourtDetector: template homography → 14 keypoints, clearance hulls; `compute_H_from_kps` |
| `objects_detector.py` | ObjectsDetector: single predict() call for person/racket/ball |
| `tracker.py` | `Tracker` (generic SORT, Kalman+Hungarian) + `BallTracker` (prefilter + gap fill) |
| `utils.py` | Video I/O, `save_coco` / `load_detections`, `text_params`; fast metadata reader `_read_meta` |
| `train_ball.py` | Fine-tune YOLO ball/object detector |
| `train_court.py` | Fine-tune court keypoint detector |
| `eval_ball.py` | Evaluate ball detector with per-class AP metrics |
| `debug_court.py` | Save intermediate court detection images for debugging |

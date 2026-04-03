# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Computer vision pipeline that analyzes tennis match video: detects players, rackets, and ball (YOLOv26x), detects court keypoints (template homography), assigns stable player IDs, filters the real tennis ball by trajectory, and outputs an annotated video.

## Running the Pipeline

```bash
python main.py
```

- **Input:** `input_videos/input_video.mp4`
- **Output:** `output_videos/output_video.mp4` (HEVC/H.265, crf=18)
- **Required model:** `models/yolo26x.pt` — single model for all three classes
- **Python env:** `.venv/bin/python`

To skip slow re-inference, set `read_from_stub=True` in `main.py` and use cached detections from:
- `tracker_stubs/player_racket_ball_detections.pkl`

Default is `read_from_stub=False` (always re-run inference on new input video).

## Dependencies

```bash
pip install ultralytics torch pandas numpy opencv-python
```

## Architecture

### Pipeline Flow (`main.py`)

```
read_video()
  → TemplateHomographyDetector (frame 0 only) — 14 court keypoints
  → YOLODetector.detect_frames() — single model.predict() for person/racket/ball
  → PlayerTracker.choose_and_filter_players() — pick 4 players, stable IDs via Kalman + Hungarian
  → RacketTracker.assign_rackets_to_players() — nearest-player assignment
  → BallTracker.filter_static_detections() — remove stationary false positives
  → BallTracker.find_ball_by_longest_trajectory() — greedy tracklet, select by movement distance
  → draw: court keypoints → players → rackets → ball → frame number
  → save_video() — ffmpeg HEVC encoding
```

### Key Design Decisions

- **Single inference:** `model.predict()` (not `model.track()`) with `classes=[0, 38, 32]` detects all three classes at once. `track()` was abandoned because ByteTrack drops unconfirmed detections (box.id=None), causing ball loss.
- **Court keypoints** detected only on frame 0 via template homography — the court doesn't move.
- **Player stable IDs:** Kalman filter predicts position, Hungarian algorithm matches detections, appearance histogram (HSV 16×16) provides ReID. 4 players selected by proximity to court keypoints.
- **Ball selection:** First removes detections stationary for ≥10 frames (false positives), then builds greedy tracklets and selects the one with greatest total movement distance.
- **Output encoding:** ffmpeg libx265, crf=18, preset=fast, tag=hvc1 (required for QuickTime compatibility).
- **Inference device:** `device='cpu'` by default. MPS tested but slower than CPU for yolo26x on M5.

### Module Responsibilities

| Module | Purpose |
|---|---|
| `trackers/detector.py` | YOLODetector: single predict() call, stub caching, MPS batch inference |
| `trackers/player_tracker.py` | Player filtering, stable ID assignment (Kalman + Hungarian + histogram) |
| `trackers/ball_tracker.py` | Static false positive filter, longest-trajectory selection |
| `trackers/racket_tracker.py` | Racket-to-player assignment by nearest center distance |
| `court_line_detector/` | TemplateHomographyDetector → 14 court keypoints (28 floats) |
| `utils/video_utils.py` | read_video, save_video (ffmpeg HEVC) |
| `utils/bbox_utils.py` | Geometric helpers (center, distance) |
| `constants/` | Player colors per ID |

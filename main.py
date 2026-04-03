import time
import cv2
import numpy as np
from utils import read_video, save_video, get_text_params
from trackers import YOLODetector, PlayerTracker, BallTracker, RacketTracker
from court_line_detector import TemplateHomographyDetector


def main():
    video_frames, video_fps = read_video("input_videos/input_video.mp4")

    court_line_detector = TemplateHomographyDetector()
    court_keypoints = court_line_detector.predict(video_frames[0])
    valid_hull = court_line_detector.get_valid_zone_hull(video_frames[0].shape)
    p_far  = np.array(court_keypoints[0:2])   # 远底线左角
    p_near = np.array(court_keypoints[4:6])   # 近底线左角
    px_per_meter = np.linalg.norm(p_near - p_far) / 23.77
    ball_max_dist = int(150 * 1000 / 3600 / video_fps * px_per_meter) + 1
    print(f"[    ball] px/m={px_per_meter:.1f}  max_dist={ball_max_dist}px")

    # YOLO 推断
    detector = YOLODetector(model_paths={
        1920: 'models/yolo26x-1920.mlpackage',
        320: 'models/yolo26x-320.mlpackage',
        160: 'models/yolo26x-160.mlpackage',
    }, imgsz=1920)
    player_detections, racket_detections, ball_detections = detector.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path="tracker_stubs/player_racket_ball_detections.pkl"
    )

    # 球拍 patch 补充
#    racket_detections = detector.augment_rackets(video_frames, player_detections, racket_detections, conf=0.05, read_from_stub=True, stub_path="tracker_stubs/racket_augmented.pkl")

    # 网球前向/后向追踪
    #ball_detections = detector.augment_balls(video_frames, ball_detections, max_dist=ball_max_dist, conf=0.05)
    #ball_detections = detector.augment_balls_backward(video_frames, ball_detections, max_dist=ball_max_dist, conf=0.05)

    # 基于 3D 三棱柱投影凸包过滤原始检测
    player_detections, racket_detections, ball_detections = detector.filter_detections(
        player_detections, racket_detections, ball_detections, valid_hull)

    # 球员筛选 + 稳定 ID（内部打印进度）
    player_tracker = PlayerTracker()
    player_detections = player_tracker.select_and_track_players(valid_hull, player_detections, video_frames)

    # 球拍归属 + 丢弃无主球拍
    racket_tracker = RacketTracker()
    racket_assignments = racket_tracker.assign_rackets_to_players(player_detections, racket_detections)

    ball_tracker = BallTracker()
    ball_tracker.draw_tracklets(video_frames, ball_detections, max_dist=ball_max_dist)
#    ball_detections = ball_tracker.find_rally(ball_detections, video_fps, max_dist=ball_max_dist)

    # 绘制
    total = len(video_frames)
    w = len(str(total))
    t0 = time.time()
    for i, frame in enumerate(video_frames):
#        cv2.polylines(frame, [valid_hull], True, (0, 255, 255), 2)
#        court_line_detector.draw_frame(frame)
#        player_tracker.draw_bboxes_frame(frame, player_detections[i])
#        racket_tracker.draw_bboxes_frame(frame, racket_assignments[i])
        ball_tracker.draw_bboxes_frame(frame, ball_detections[i])
        fs, ft = get_text_params(frame.shape[0], base_height=1080)
        margin = int(frame.shape[0] * 0.028)  # ~30px @ 1080p
        cv2.putText(frame, str(i), (margin, frame.shape[0] - margin), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.5, (0, 255, 0), ft)
        pct = (i + 1) * 100 // total
        print(f"[    draw] {i+1:>{w}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
    print(f"[    draw] {total:>{w}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")

    save_video(video_frames, "output_videos/output_video.mp4", fps=video_fps)


if __name__ == "__main__":
    main()

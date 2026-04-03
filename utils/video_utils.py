import cv2
import os
import time

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps

def save_video(output_video_frames, output_video_path, fps=24):
    h, w = output_video_frames[0].shape[:2]
    final_path = os.path.splitext(output_video_path)[0] + '.mp4'
    out = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    total = len(output_video_frames)
    fw = len(str(total))
    t0 = time.time()
    for i, frame in enumerate(output_video_frames):
        out.write(frame)
        pct = (i + 1) * 100 // total
        print(f"[   video] {i+1:>{fw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
    out.release()
    elapsed = time.time() - t0
    print(f"[   video] {total:>{fw}}/{total} frames  (100%)  done: {elapsed:>6.1f}s")
    print(f"[   video] saved → {final_path}", flush=True)
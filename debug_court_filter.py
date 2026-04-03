"""调试脚本：将球场凸包和 player margin 区域画在第 0 帧并保存。"""
import cv2
import numpy as np
from utils import read_video
from court_line_detector import TemplateHomographyDetector


def draw_court_filter(frame, court_keypoints, player_margin=0.15, ball_margin=0.05):
    pts = np.array([(court_keypoints[i*2], court_keypoints[i*2+1])
                    for i in range(len(court_keypoints) // 2)], dtype=np.float32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2)).astype(np.int32)

    court_h = pts[:, 1].max() - pts[:, 1].min()
    player_px = int(court_h * player_margin)
    ball_px   = int(court_h * ball_margin)

    # 在 mask 上绘制凸包，再膨胀 player_px 得到 margin 区域
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    kernel_p = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (player_px*2+1, player_px*2+1))
    margin_mask_p = cv2.dilate(mask, kernel_p)

    kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ball_px*2+1, ball_px*2+1))
    margin_mask_b = cv2.dilate(mask, kernel_b)

    out = frame.copy()

    # 球场 margin 区域（player）：蓝色半透明
    only_margin_p = cv2.bitwise_and(margin_mask_p, cv2.bitwise_not(mask))
    overlay = out.copy()
    overlay[only_margin_p > 0] = (200, 100, 0)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    # 球场内部：绿色半透明
    overlay2 = out.copy()
    overlay2[mask > 0] = (0, 180, 0)
    cv2.addWeighted(overlay2, 0.25, out, 0.75, 0, out)

    # 凸包轮廓
    cv2.polylines(out, [hull], True, (0, 255, 0), 2)

    # margin 外边界（player）
    contours_p, _ = cv2.findContours(margin_mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours_p, -1, (200, 100, 0), 2)

    # margin 外边界（ball）
    contours_b, _ = cv2.findContours(margin_mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours_b, -1, (0, 200, 255), 2)

    # 关键点
    for i in range(len(court_keypoints) // 2):
        x, y = int(court_keypoints[i*2]), int(court_keypoints[i*2+1])
        cv2.circle(out, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(out, str(i), (x+8, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 图例
    h = frame.shape[0]
    s = max(1, h // 1080)
    cv2.putText(out, "Green: court hull",           (20, h-120*s), cv2.FONT_HERSHEY_SIMPLEX, 0.7*s, (0,255,0),     2*s)
    cv2.putText(out, "Blue: player margin (15%)",   (20, h- 80*s), cv2.FONT_HERSHEY_SIMPLEX, 0.7*s, (200,100,0),   2*s)
    cv2.putText(out, "Cyan: ball margin (5%)",      (20, h- 40*s), cv2.FONT_HERSHEY_SIMPLEX, 0.7*s, (0,200,255),   2*s)

    return out


if __name__ == "__main__":
    frames, _ = read_video("input_videos/input_video.mp4")
    detector = TemplateHomographyDetector()
    keypoints = detector.predict(frames[0])
    result = draw_court_filter(frames[0], keypoints)
    cv2.imwrite("output_videos/debug_court_filter.jpg", result)
    print("saved → output_videos/debug_court_filter.jpg")

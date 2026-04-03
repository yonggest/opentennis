import cv2
import constants


def draw_player_stats(output_video_frames, player_stats):
    player_ids = sorted({
        int(col.split('_')[1])
        for col in player_stats.columns
        if col.startswith('player_') and col.endswith('_number_of_shots')
    })

    label_w = 75
    val_w   = 85
    row_h   = 22
    pad_x   = 10
    pad_y   = 20
    n = len(player_ids)
    box_w = pad_x * 2 + label_w + val_w * n
    box_h = pad_y + row_h * 5   # header + 4 rows

    labels = [
        ("Shot(km/h)",   "last_shot_speed"),
        ("Move(km/h)",   "last_player_speed"),
        ("AvgShot",      "average_shot_speed"),
        ("AvgMove",      "average_player_speed"),
    ]

    for index, row in player_stats.iterrows():
        frame = output_video_frames[index]

        start_x = frame.shape[1] - box_w - 15
        start_y = frame.shape[0] - box_h - 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y),
                      (start_x + box_w, start_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        def put(text, x, y, color=(255, 255, 255), scale=0.38, thickness=1):
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

        # 表头：球员ID（带颜色）
        for ci, pid in enumerate(player_ids):
            color = constants.PLAYER_COLORS.get(pid, (255, 255, 255))
            x = start_x + pad_x + label_w + ci * val_w
            y = start_y + pad_y
            put(f"P{pid}", x, y, color=color, scale=0.45, thickness=2)

        # 数据行
        for ri, (label, key) in enumerate(labels, 1):
            y = start_y + pad_y + ri * row_h
            put(label, start_x + pad_x, y)
            for ci, pid in enumerate(player_ids):
                val = row.get(f'player_{pid}_{key}', 0)
                if val != val:
                    val = 0.0
                x = start_x + pad_x + label_w + ci * val_w
                put(f"{val:.1f}", x, y)

        output_video_frames[index] = frame

    return output_video_frames

import cv2
import os

result_dir = "/home/multy-surya/Grounded-SAM-2/runs/vrx_yolo11/buoy_det2_vis"
output_video = "/home/multy-surya/Grounded-SAM-2/runs/vrx_yolo11/yolo_vrx_output.mp4"
fps = 15

# 获取所有 result 帧
frames = sorted(
    [f for f in os.listdir(result_dir) if f.endswith(".jpg")],
    key=lambda x: int(os.path.splitext(x)[0])
)

if len(frames) == 0:
    print("No frames found!")
    exit()

# 读取第一帧确定长宽
first_frame = cv2.imread(os.path.join(result_dir, frames[0]))
h, w, _ = first_frame.shape

# 初始化 video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# 写入每一帧
for f in frames:
    img = cv2.imread(os.path.join(result_dir, f))
    video_writer.write(img)

video_writer.release()
print("Video saved to:", output_video)

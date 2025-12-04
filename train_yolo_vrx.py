from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")  # 或 "yolo11s.pt"

    model.train(
        data="vrx_buoy.yaml",
        epochs=50,
        imgsz=1280,
        batch=16,
        device=0,
        workers=4,
        project="runs/vrx_yolo11",
        name="buoy_det"
    )

    # 训练完自动用 best 模型验证一下
    model.val()

if __name__ == "__main__":
    main()

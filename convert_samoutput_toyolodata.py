import os
import json
import shutil
import random
import cv2
import numpy as np

# ========= 路径配置 =========
VIDEO_DIR = "vrx_demo"                          # 原始图片所在目录
GT_JSON_DIR = "vrx_manual_gt_outputs/json_data" # SAM2 GT json
GT_MASK_DIR = "vrx_manual_gt_outputs/mask_data" # SAM2 mask npy
YOLO_ROOT = "vrx_yolo11_dataset"                # 新建数据集根目录

IMG_OUT_DIR = os.path.join(YOLO_ROOT, "images")
LBL_OUT_DIR = os.path.join(YOLO_ROOT, "labels")

os.makedirs(os.path.join(IMG_OUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(IMG_OUT_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(LBL_OUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(LBL_OUT_DIR, "val"), exist_ok=True)

# ========= 语义类别 → YOLO class id =========
CLASS_NAME_TO_ID = {
    "boat": 0,
    "red_cone_buoy": 1,
    "black_cone_buoy": 2,
    "green_cone_buoy": 3,
    "gray_cone_buoy": 4,
    "orange_sphere_buoy": 5,
    "black_sphere_buoy": 6,
}

# ========= SAM2 实例名 → 语义类别名 =========
# 你刚才给出的映射（我已经翻转好了）：
OBJECT_TO_CLASS = {
    "object_1": "boat",
    "object_2": "red_cone_buoy",
    "object_3": "black_cone_buoy",
    "object_4": "green_cone_buoy",
    "object_5": "gray_cone_buoy",
    "object_6": "orange_sphere_buoy",
    "object_7": "black_sphere_buoy",
    # 如果 object_8 也有类别，就在这里补：
    # "object_8": "xxx",
}


def split_train_val(all_items, val_ratio=0.2):
    random.shuffle(all_items)
    n_val = int(len(all_items) * val_ratio)
    val_items = set(all_items[:n_val])
    train_items = set(all_items[n_val:])
    return train_items, val_items


def mask_to_box(mask_bool):
    """
    从一个 bool mask 计算 [x1, y1, x2, y2]（像素坐标）
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    return [int(x1), int(y1), int(x2), int(y2)]


def main():
    json_files = sorted([f for f in os.listdir(GT_JSON_DIR) if f.endswith(".json")])
    base_names = [f.replace("mask_", "").replace(".json", "") for f in json_files]

    if len(base_names) == 0:
        print("在 json_data 目录下没有找到任何 json 文件，检查路径是否正确。")
        return

    train_set, val_set = split_train_val(base_names, val_ratio=0.2)

    for base in base_names:
        json_path = os.path.join(GT_JSON_DIR, f"mask_{base}.json")
        mask_path = os.path.join(GT_MASK_DIR, f"mask_{base}.npy")

        if not os.path.exists(json_path):
            print("找不到 json:", json_path)
            continue
        if not os.path.exists(mask_path):
            print("找不到 mask npy:", mask_path)
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        # 读取 mask 用于必要时计算 bbox
        mask_array = np.load(mask_path)  # H × W, 每个像素是 instance_id (int)
        H = data.get("mask_height", mask_array.shape[0])
        W = data.get("mask_width", mask_array.shape[1])

        img_name_jpg = f"{base}.jpg"
        img_path = os.path.join(VIDEO_DIR, img_name_jpg)
        img = cv2.imread(img_path)
        if img is None:
            print("无法读取图片:", img_path)
            continue

        subset = "train" if base in train_set else "val"
        out_img_path = os.path.join(IMG_OUT_DIR, subset, img_name_jpg)
        out_lbl_path = os.path.join(LBL_OUT_DIR, subset, img_name_jpg.replace(".jpg", ".txt"))

        # 复制图片
        shutil.copy(img_path, out_img_path)

        # 写 label
        with open(out_lbl_path, "w") as lf:
            # 假设 data["labels"] 是一个 dict: { "1": {...}, "2": {...}, ... }
            for obj_id_str, obj_info in data["labels"].items():
                cls_name_raw = obj_info["class_name"]  # e.g. "object_2"

                # step1: object_x → 语义类别名
                if cls_name_raw not in OBJECT_TO_CLASS:
                    print(f"[警告] OBJECT_TO_CLASS 中没有映射 {cls_name_raw}，跳过该实例。")
                    continue
                cls_name = OBJECT_TO_CLASS[cls_name_raw]  # e.g. "red_cone_buoy"

                # step2: 语义类别名 → YOLO class id
                if cls_name not in CLASS_NAME_TO_ID:
                    print(f"[警告] CLASS_NAME_TO_ID 中没有类别 {cls_name}，跳过该实例。")
                    continue
                cls_id = CLASS_NAME_TO_ID[cls_name]

                # step3: 拿 bbox
                if "box" in obj_info:
                    x1, y1, x2, y2 = obj_info["box"]
                else:
                    # 如果 json 里没有 box，就从 mask_array 里根据 instance_id 计算
                    try:
                        inst_id = int(obj_id_str)
                    except ValueError:
                        # 有些实现里 instance_id 单独存，且 obj_id_str 不是数字
                        inst_id = obj_info.get("instance_id", None)
                    if inst_id is None:
                        print(f"[警告] {base} 中 {obj_id_str} 没有 instance_id，且无 box，跳过。")
                        continue

                    mask_bool = (mask_array == inst_id)
                    box = mask_to_box(mask_bool)
                    if box is None:
                        print(f"[警告] {base} 中 {obj_id_str} 的 mask 为空，跳过。")
                        continue
                    x1, y1, x2, y2 = box

                # step4: 转为 YOLO 格式（归一化）
                xc = (x1 + x2) / 2.0 / W
                yc = (y1 + y2) / 2.0 / H
                ww = (x2 - x1) / float(W)
                hh = (y2 - y1) / float(H)

                lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

    print("✅ 转换完成，数据集保存在:", YOLO_ROOT)


if __name__ == "__main__":
    main()

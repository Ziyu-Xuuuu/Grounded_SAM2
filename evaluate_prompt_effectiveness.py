import os
import numpy as np
import json
import pandas as pd
from tqdm import tqdm

# -----------------------------
# 配置路径（替换成你自己的）
# -----------------------------
gt_dir = "vrx_manual_gt_outputs/mask_data"
gt_json_dir = "vrx_manual_gt_outputs/json_data"
pred_dir = "vrx_demo_outputs/mask_data"
pred_json_dir = "vrx_demo_outputs/json_data"

output_csv = "prompt_evaluation.csv"

# -----------------------------
# IoU计算函数
# -----------------------------
def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def compute_precision(gt_mask, pred_mask):
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fp = np.logical_and(~gt_mask, pred_mask).sum()
    return tp / (tp + fp + 1e-6)

def compute_recall(gt_mask, pred_mask):
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fn = np.logical_and(gt_mask, ~pred_mask).sum()
    return tp / (tp + fn + 1e-6)

# -----------------------------
# 遍历所有帧
# -----------------------------
records = []

frame_ids = sorted([f for f in os.listdir(gt_dir) if f.endswith(".npy")])

for fname in tqdm(frame_ids, desc="Evaluating"):
    frame_id = fname.replace("mask_", "").replace(".npy", "")

    gt_path = os.path.join(gt_dir, fname)
    pred_path = os.path.join(pred_dir, fname)

    if not os.path.exists(pred_path):
        continue
    
    gt_mask = np.load(gt_path)
    pred_mask = np.load(pred_path)

    # 读取 json 获取 object class 和 prompt label
    gt_json = json.load(open(os.path.join(gt_json_dir, fname.replace(".npy",".json"))))
    pred_json = json.load(open(os.path.join(pred_json_dir, fname.replace(".npy",".json"))))

    gt_objects = gt_json["labels"]
    pred_objects = pred_json["labels"]

    # 遍历所有 GT 物体
    for obj_id in gt_objects:
        gt_obj_mask = (gt_mask == int(obj_id))

        # 自动预测没有这个 id（FN）
        if str(obj_id) not in pred_objects:
            records.append({
                "frame": frame_id,
                "object_id": obj_id,
                "prompt_class": gt_objects[obj_id]["class_name"],
                "iou": 0,
                "precision": 0,
                "recall": 0,
                "detected": False
            })
            continue

        pred_obj_mask = (pred_mask == int(obj_id))

        iou = compute_iou(gt_obj_mask, pred_obj_mask)
        precision = compute_precision(gt_obj_mask, pred_obj_mask)
        recall = compute_recall(gt_obj_mask, pred_obj_mask)

        records.append({
            "frame": frame_id,
            "object_id": obj_id,
            "prompt_class": gt_objects[obj_id]["class_name"],
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "detected": True
        })

# 存储结果
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print("保存至：", output_csv)

import os
import cv2
import torch
import numpy as np
from PIL import Image
import copy
import sys
from pathlib import Path

# 找到 Grounded-SAM-2 的根目录：.../Grounded-SAM-2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("DEBUG PROJECT_ROOT:", PROJECT_ROOT)

import sam2
print("DEBUG sam2 loaded from:", sam2.__file__)
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

"""
SAM2 手动标注 + 视频 tracking 生成 GT 的脚本

使用方式（默认只在第0帧上标注）：
- 左键：添加正样本点（foreground = 1）
- 右键：添加负样本点（background = 0，可选）
- 键盘 'n'：根据当前点集生成一个新的实例 mask（object_k）
- 键盘 'z'：撤销当前实例里最后一个点
- 键盘 'c'：清空当前正在编辑的实例的点
- 键盘 'q'：结束标注，开始整段视频 tracking

注意：
1. 这里只做一次关键帧（frame 0）的人工标注并向后 tracking，
   你可以仿照这个逻辑扩展到多个关键帧或双向 tracking。
2. 依赖你的 MaskDictionaryModel 和 CommonUtils，与自动 GroundingDINO 版本保持兼容。
"""

# =============== 可配置参数 ===============
video_dir = "vrx_demo"                     # 帧目录, 文件名为 0.jpg,1.jpg,...
output_dir = "vrx_manual_gt_outputs"      # 输出目录
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

key_frame_idx = 0                          # 选择哪一帧做手动标注
frame_step = 1                             # tracking 步长（1 = 每帧）
PROMPT_TYPE_FOR_VIDEO = "mask"
# =======================================


# ---------- 鼠标交互状态 ----------
class AnnotationState:
    def __init__(self):
        # 当前正在编辑的实例的点
        self.current_pos_points = []  # [(x,y), ...]
        self.current_neg_points = []
        # 已经完成的实例
        self.instances = []  # 每个元素：dict{ 'pos': [...], 'neg': [...], 'mask': np.ndarray or None }
        self.window_name = "SAM2 Manual Annotation"
        self.image_vis = None
        self.base_image = None

    def reset_current_points(self):
        self.current_pos_points = []
        self.current_neg_points = []


ann_state = AnnotationState()


def draw_points_on_image():
    """
    根据当前状态，在 ann_state.base_image 上画出标注点和实例轮廓
    """
    if ann_state.base_image is None:
        return

    img = ann_state.base_image.copy()

    # 先画已经完成的实例轮廓（如果有 mask）
    for idx, inst in enumerate(ann_state.instances, start=1):
        if 'mask' in inst and inst['mask'] is not None:
            mask = inst['mask'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            # 在轮廓附近写上id
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(img, f"ID {idx}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 再画当前实例的点
    for (x, y) in ann_state.current_pos_points:
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)  # 正样本点: 红色
    for (x, y) in ann_state.current_neg_points:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)  # 负样本点: 蓝色

    ann_state.image_vis = img
    cv2.imshow(ann_state.window_name, ann_state.image_vis)


def mouse_callback(event, x, y, flags, param):
    """
    鼠标回调：左键正样本，右键负样本
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        ann_state.current_pos_points.append((x, y))
        draw_points_on_image()
    elif event == cv2.EVENT_RBUTTONDOWN:
        ann_state.current_neg_points.append((x, y))
        draw_points_on_image()


def manual_annotate_key_frame(image_bgr, image_predictor: SAM2ImagePredictor):
    """
    对关键帧进行手动标注，返回一个 MaskDictionaryModel
    image_bgr: OpenCV BGR 图像
    """
    h, w = image_bgr.shape[:2]
    ann_state.base_image = image_bgr.copy()
    ann_state.image_vis = image_bgr.copy()
    ann_state.instances = []
    ann_state.reset_current_points()

    cv2.namedWindow(ann_state.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ann_state.window_name, mouse_callback)

    print("=== 手动标注说明 ===")
    print("左键：添加正样本点 (foreground)")
    print("右键：添加负样本点 (background，可选)")
    print("'n'：根据当前点集生成一个新的实例 mask")
    print("'z'：撤销当前实例的最后一个点")
    print("'c'：清空当前实例的点")
    print("'q'：结束标注并生成 GT")
    print("===================")

    image_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    draw_points_on_image()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('n'):
            # 生成一个新实例的 mask
            if len(ann_state.current_pos_points) == 0:
                print("当前实例没有正样本点，无法生成 mask。")
                continue

            point_coords = ann_state.current_pos_points + ann_state.current_neg_points
            point_labels = [1] * len(ann_state.current_pos_points) + [0] * len(ann_state.current_neg_points)

            point_coords_np = np.array(point_coords, dtype=np.float32)
            point_labels_np = np.array(point_labels, dtype=np.int32)

            print(f"使用 {len(ann_state.current_pos_points)} 个正样本点, "
                  f"{len(ann_state.current_neg_points)} 个负样本点 生成实例 {len(ann_state.instances)+1}")

            with torch.no_grad():
                masks, scores, logits = image_predictor.predict(
                    point_coords=point_coords_np,
                    point_labels=point_labels_np,
                    multimask_output=True  # 让SAM2给多种候选
                )

            # 选择得分最高的候选 mask
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]  # (1,H,W) 或 (H,W)
            if best_mask.ndim == 3:
                best_mask = best_mask[0]

            inst = {
                "pos": ann_state.current_pos_points.copy(),
                "neg": ann_state.current_neg_points.copy(),
                "mask": (best_mask > 0)  # bool
            }
            ann_state.instances.append(inst)
            print(f"实例 {len(ann_state.instances)} 生成完成。")

            # 清空当前实例点集，准备标注下一个实例
            ann_state.reset_current_points()
            draw_points_on_image()

        elif key == ord('z'):
            # 撤销当前实例的最后一个点（优先正样本）
            if len(ann_state.current_pos_points) > 0:
                ann_state.current_pos_points.pop()
            elif len(ann_state.current_neg_points) > 0:
                ann_state.current_neg_points.pop()
            draw_points_on_image()

        elif key == ord('c'):
            # 清空当前实例点
            ann_state.reset_current_points()
            draw_points_on_image()

        elif key == ord('q'):
            print("结束标注。")
            break

        elif key == 27:  # ESC
            print("按 ESC 退出且丢弃所有标注。")
            ann_state.instances = []
            break

    cv2.destroyWindow(ann_state.window_name)

    # 把实例写入 MaskDictionaryModel
    mask_dict = MaskDictionaryModel(
        promote_type=PROMPT_TYPE_FOR_VIDEO,
        mask_name=f"mask_{key_frame_idx}.npy"  # 稍后在保存时再按真实帧名改
    )
    mask_dict.mask_height = h
    mask_dict.mask_width = w

    for idx, inst in enumerate(ann_state.instances, start=1):
        mask = inst["mask"]
        obj = ObjectInfo(
            instance_id=idx,
            mask=torch.tensor(mask, dtype=torch.bool),
            class_name=f"object_{idx}",
            logit=None
        )
        obj.update_box()
        mask_dict.labels[idx] = obj

    print(f"总共标注了 {len(mask_dict.labels)} 个实例。")
    return mask_dict


def main():
    # ========== 准备环境 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 创建输出目录
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    # 读取所有帧名
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    print("Total frames:", len(frame_names))
    if len(frame_names) == 0:
        print("No frames found in", video_dir)
        return

    # 加载 SAM2 模型
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # 初始化 video predictor
    inference_state = video_predictor.init_state(video_path=video_dir)

    # ========== Step 1: 在关键帧上做手动标注 ==========
    key_frame_name = frame_names[key_frame_idx]
    key_frame_path = os.path.join(video_dir, key_frame_name)
    key_frame_bgr = cv2.imread(key_frame_path)
    if key_frame_bgr is None:
        print("Failed to read key frame:", key_frame_path)
        return

    mask_dict_key = manual_annotate_key_frame(key_frame_bgr, image_predictor)

    if len(mask_dict_key.labels) == 0:
        print("没有标注任何实例，退出。")
        return

    # 把 mask_dict_key 的 mask_name 换成真实名字
    key_base_name = os.path.splitext(key_frame_name)[0]
    mask_dict_key.mask_name = f"mask_{key_base_name}.npy"

    # ========== Step 2: 从关键帧向前/向后 tracking ==========
    # 这里只演示向后 tracking（关键帧 -> 最后一帧）
    print("开始视频 tracking...")
    objects_count = len(mask_dict_key.labels)

    # 把关键帧的 mask 先保存下来
    mask_img = torch.zeros(mask_dict_key.mask_height, mask_dict_key.mask_width, dtype=torch.int32)
    for obj_id, obj_info in mask_dict_key.labels.items():
        mask_img[obj_info.mask == True] = obj_id
    mask_np = mask_img.numpy().astype(np.uint16)
    np.save(os.path.join(mask_data_dir, mask_dict_key.mask_name), mask_np)

    json_data_path = os.path.join(json_data_dir, mask_dict_key.mask_name.replace(".npy", ".json"))
    mask_dict_key.to_json(json_data_path)

    # video_predictor 开始 state
    video_predictor.reset_state(inference_state)

    # 把关键帧上的每个实例注册进 video_predictor
    for object_id, object_info in mask_dict_key.labels.items():
        video_predictor.add_new_mask(
            inference_state,
            key_frame_idx,
            object_id,
            object_info.mask  # bool 2D
        )

    # 用 propagate_in_video 往后 tracking
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
        inference_state,
        max_frame_num_to_track=len(frame_names) - key_frame_idx - 1,
        start_frame_idx=key_frame_idx
    ):
        frame_masks = MaskDictionaryModel()
        frame_base_name = os.path.splitext(frame_names[out_frame_idx])[0]

        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0)  # (1,H,W) bool
            if out_mask.ndim == 3:
                out_mask = out_mask[0]

            if out_mask.sum() == 0:
                print(f"warning: empty mask for object {out_obj_id} at frame {out_frame_idx}")
                continue

            class_name = mask_dict_key.get_target_class_name(out_obj_id)
            obj = ObjectInfo(
                instance_id=out_obj_id,
                mask=out_mask,
                class_name=class_name,
                logit=mask_dict_key.get_target_logit(out_obj_id)
            )
            obj.update_box()
            frame_masks.labels[out_obj_id] = obj

        if len(frame_masks.labels) == 0:
            # 该帧所有实例都丢了
            continue

        frame_masks.mask_name = f"mask_{frame_base_name}.npy"
        frame_masks.mask_height = out_mask.shape[-2]
        frame_masks.mask_width = out_mask.shape[-1]
        video_segments[out_frame_idx] = frame_masks

    print("video_segments:", len(video_segments))

    # ========== Step 3: 保存 tracking 结果 ==========
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        frame_masks_info.to_json(json_data_path)

    # ========== Step 4: 可视化检查 ==========
    CommonUtils.draw_masks_and_box_with_supervision(
        video_dir, mask_data_dir, json_data_dir, result_dir
    )
    print("GT 生成完成，结果保存在：", output_dir)


if __name__ == "__main__":
    main()

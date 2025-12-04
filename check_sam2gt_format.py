import os
import json
import numpy as np

JSON_DIR = "vrx_manual_gt_outputs/json_data"
MASK_DIR = "vrx_manual_gt_outputs/mask_data"

def mask_to_box(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return [xs.min(), ys.min(), xs.max(), ys.max()]

def main():
    json_files = sorted([f for f in os.listdir(JSON_DIR) if f.endswith(".json")])

    for jf in json_files:
        base = jf.replace("mask_", "").replace(".json", "")
        json_path = os.path.join(JSON_DIR, jf)
        mask_path = os.path.join(MASK_DIR, f"mask_{base}.npy")

        print(f"\n=== 检查: {jf} ===")

        if not os.path.exists(mask_path):
            print(f"❌ 对应 mask 文件不存在: {mask_path}")
            continue

        mask_array = np.load(mask_path)
        with open(json_path, "r") as f:
            data = json.load(f)

        labels = data.get("labels", {})
        if not labels:
            print("⚠️  labels 字段为空")
            continue

        for obj_id_str, obj_info in labels.items():
            print(f"--- 实例 {obj_id_str} ---")

            # 1) JSON 是否有 box？
            if "box" in obj_info:
                print(f"   ✅ JSON 中已有 box: {obj_info['box']}")
            else:
                print("   ❌ JSON 中缺少 box")

                # 2) 试图从 mask 计算
                inst_id = obj_info.get("instance_id", None)
                if inst_id is None:
                    print("   ❌ instance_id 缺失，无法从 mask 定位")
                    continue

                mask_bool = (mask_array == int(inst_id))
                box = mask_to_box(mask_bool)

                if box is None:
                    print("   ❌ mask 中没有找到对应 instance_id 的像素")
                else:
                    print(f"   ✔ 可以从 mask 计算出 bbox: {box}")


if __name__ == "__main__":
    main()

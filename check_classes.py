import os
import json

GT_JSON_DIR = "vrx_manual_gt_outputs/json_data"

classes = set()

for fname in os.listdir(GT_JSON_DIR):
    if not fname.endswith(".json"):
        continue
    json_path = os.path.join(GT_JSON_DIR, fname)

    with open(json_path, "r") as f:
        data = json.load(f)

    for obj_id, obj in data["labels"].items():
        classes.add(obj["class_name"])

print("=== 所有出现过的 class_name ===")
for c in sorted(classes):
    print(c)

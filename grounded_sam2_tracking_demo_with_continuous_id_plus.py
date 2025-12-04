import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
import pandas as pd
import matplotlib.pyplot as plt

# This demo shows the continuous object tracking plus reverse tracking with Grounding DINO and SAM 2
"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = (
    "red cone buoy. green cone buoy. gray cone buoy. black cone buoy. "
    "black sphere buoy. orange sphere buoy. "
    "red buoy. green buoy. gray buoy. black buoy. orange buoy. "
    "cone buoy. sphere buoy. cylindrical buoy. "
    "navigation buoy. marker buoy. floating buoy. marine buoy. "
    "red navigation marker. green navigation marker. black navigation marker. "
    "floating marker. marine marker. water marker. "
    "buoy. marker. float. navigation aid. "
    "red float. green float. black float. orange float. "
    "marine equipment. navigation equipment. floating object. "
    "boat. ship. vessel. "
    "dock. pier. rocks. water. lake. sea. sky."
)

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
video_dir = "vrx_demo"
# 'output_dir' is the directory to save the annotated frames
output_dir = "vrx_demo_outputs"
# 'output_video_path' is the path to save the final video
output_video_path = "./vrx_demo_outputs/vrx_tracking_demo.mp4"
# create the output directory
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)
step = 20 # the step to sample frames for Grounding DINO predictor

# Initialize confidence tracking
confidence_data = []  # Store confidence data over time

sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0
frame_object_count = {}
"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("Total frames:", len(frame_names))
for start_frame_idx in range(0, len(frame_names), step):
# prompt grounding dino to get the box coordinates on specific frame
    print("start_frame_idx", start_frame_idx)
    # continue
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path).convert("RGB")
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"] # .cpu().numpy()
    detection_scores = results[0]["scores"]  # Get detection confidence scores
    # print("results[0]",results[0])
    OBJECTS = results[0]["labels"]
    if input_boxes.shape[0] != 0:

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        """
        Step 3: Register each object's positive points to video predictor
        """

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
            
            # Collect confidence data for initial detection
            for i, (det_score, sam_score, obj_label) in enumerate(zip(detection_scores, scores, OBJECTS)):
                confidence_data.append({
                    'frame_idx': start_frame_idx,
                    'object_id': objects_count + i + 1,
                    'object_class': obj_label,
                    'detection_confidence': float(det_score),
                    'sam_confidence': float(sam_score),
                    'tracking_type': 'initial_detection'
                })
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")
    else:
        print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
        mask_dict = sam2_masks

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
    frame_object_count[start_frame_idx] = objects_count
    print("objects_count", objects_count)
    
    if len(mask_dict.labels) == 0:
        mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[start_frame_idx:start_frame_idx+step])
        print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
        continue
    else:
        video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )
        
        video_segments = {}  # output the following {step} frames tracking masks
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
            frame_masks = MaskDictionaryModel()
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                mask_confidence = torch.max(out_mask_logits[i]).item()  # Get max logit as confidence
                object_class = mask_dict.get_target_class_name(out_obj_id)
                
                if out_mask.sum() == 0:
                    print(f"no mask for object {out_obj_id} ({object_class}) at frame {out_frame_idx}")
                    continue
                    
                object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = object_class, logit=mask_dict.get_target_logit(out_obj_id))
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]
                
                # Collect confidence data for tracking
                confidence_data.append({
                    'frame_idx': out_frame_idx,
                    'object_id': out_obj_id,
                    'object_class': object_class,
                    'mask_confidence': mask_confidence,
                    'mask_area': int(out_mask.sum()),
                    'tracking_type': 'propagation'
                })

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        print("video_segments:", len(video_segments))
    """
    Step 5: save the tracking masks and json files
    """
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        frame_masks_info.to_json(json_data_path)
       

CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

print("try reverse tracking")
start_object_id = 0
object_info_dict = {}
for frame_idx, current_object_count in frame_object_count.items():
    print("reverse tracking frame", frame_idx, frame_names[frame_idx])
    print(f"Debug: start_object_id={start_object_id}, current_object_count={current_object_count}")
    
    masks_added = False
    if frame_idx != 0:
        video_predictor.reset_state(inference_state)
        image_base_name = frame_names[frame_idx].split(".")[0]
        json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
        json_data = MaskDictionaryModel().from_json(json_data_path)
        mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
        mask_array = np.load(mask_data_path)
        
        for object_id in range(start_object_id+1, current_object_count+1):
            object_class = json_data.labels[object_id].class_name
            print(f"reverse tracking object {object_id} ({object_class})")
            object_info_dict[object_id] = json_data.labels[object_id]
            video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
            masks_added = True
    
    start_object_id = current_object_count
        
    
    # Only propagate if we have added masks
    if masks_added:
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step*2,  start_frame_idx=frame_idx, reverse=True):
            image_base_name = frame_names[out_frame_idx].split(".")[0]
            json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
            json_data = MaskDictionaryModel().from_json(json_data_path)
            mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
            mask_array = np.load(mask_data_path)
            # merge the reverse tracking masks with the original masks
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu()
                if out_mask.sum() == 0:
                    object_class = object_info_dict[out_obj_id].class_name if out_obj_id in object_info_dict else "unknown"
                    print(f"no mask for object {out_obj_id} ({object_class}) at frame {out_frame_idx}")
                    continue
                object_info = object_info_dict[out_obj_id]
                object_info.mask = out_mask[0]
                object_info.update_box()
                json_data.labels[out_obj_id] = object_info
                mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                mask_array[object_info.mask] = out_obj_id
            
            np.save(mask_data_path, mask_array)
            json_data.to_json(json_data_path)

        



"""
Step 6: Draw the results and save the video
"""
CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir+"_reverse")

create_video_from_images(result_dir, output_video_path, frame_rate=15)

print("Analyzing confidence data...")

# Convert confidence data to DataFrame for analysis
if confidence_data:
    df = pd.DataFrame(confidence_data)
    
    # Save confidence data to CSV
    confidence_csv_path = os.path.join(output_dir, "confidence_analysis.csv")
    df.to_csv(confidence_csv_path, index=False)
    print(f"Confidence data saved to: {confidence_csv_path}")
    
    # Filter data to only show buoy objects (include all buoy-related prompts)
    buoy_classes = [
        # Original specific buoys
        "red cone buoy", "green cone buoy", "gray cone buoy", "black cone buoy", 
        "black sphere buoy", "orange sphere buoy",
        # Color-only buoys
        "red buoy", "green buoy", "gray buoy", "black buoy", "orange buoy",
        # Shape-only buoys  
        "cone buoy", "sphere buoy", "cylindrical buoy",
        # Navigation buoys
        "navigation buoy", "marker buoy", "floating buoy", "marine buoy",
        # Navigation markers
        "red navigation marker", "green navigation marker", "black navigation marker",
        "floating marker", "marine marker", "water marker",
        # Generic terms
        "buoy", "marker", "float", "navigation aid",
        # Color floats
        "red float", "green float", "black float", "orange float",
        # Equipment terms
        "marine equipment", "navigation equipment", "floating object"
    ]
    df_filtered = df[df['object_class'].isin(buoy_classes)]
    
    if len(df_filtered) == 0:
        print("No buoy objects found in confidence data.")
        df_plot = df  # Fall back to all data
        plot_title_suffix = " (All Objects - No Buoys Found)"
    else:
        df_plot = df_filtered
        plot_title_suffix = " (Buoys Only)"
        print(f"Filtering to {len(df_filtered)} records from {len(buoy_classes)} buoy classes")
    
    # Create confidence vs time visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Detection confidence over time grouped by object class
    if 'detection_confidence' in df.columns:
        plt.subplot(2, 2, 1)
        detection_df = df_plot[df_plot['tracking_type'] == 'initial_detection']
        
        # Group by object class and combine data
        for obj_class in detection_df['object_class'].unique():
            class_data = detection_df[detection_df['object_class'] == obj_class]
            # Sort by frame index to create smooth lines
            class_data = class_data.sort_values('frame_idx')
            plt.plot(class_data['frame_idx'], class_data['detection_confidence'], 
                    marker='o', label=f'{obj_class}', alpha=0.7)
            
        plt.title(f'Detection Confidence Over Time{plot_title_suffix}')
        plt.xlabel('Frame Index')
        plt.ylabel('Detection Confidence')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    
    # Plot 2: SAM confidence over time grouped by object class
    if 'sam_confidence' in df.columns:
        plt.subplot(2, 2, 2)
        sam_df = df_plot[df_plot['tracking_type'] == 'initial_detection']
        
        # Group by object class and combine data
        for obj_class in sam_df['object_class'].unique():
            class_data = sam_df[sam_df['object_class'] == obj_class]
            # Sort by frame index to create smooth lines
            class_data = class_data.sort_values('frame_idx')
            plt.plot(class_data['frame_idx'], class_data['sam_confidence'], 
                    marker='s', label=f'{obj_class}', alpha=0.7)
            
        plt.title(f'SAM Confidence Over Time{plot_title_suffix}')
        plt.xlabel('Frame Index')
        plt.ylabel('SAM Confidence')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    
    # Plot 3: Mask confidence during tracking grouped by object class
    if 'mask_confidence' in df.columns:
        plt.subplot(2, 2, 3)
        tracking_df = df_plot[df_plot['tracking_type'] == 'propagation']
        
        # Group by object class and combine data
        for obj_class in tracking_df['object_class'].unique():
            class_data = tracking_df[tracking_df['object_class'] == obj_class]
            # Sort by frame index to create smooth lines
            class_data = class_data.sort_values('frame_idx')
            plt.plot(class_data['frame_idx'], class_data['mask_confidence'], 
                    marker='.', alpha=0.6, label=f'{obj_class}')
            
        plt.title(f'Mask Confidence During Tracking{plot_title_suffix}')
        plt.xlabel('Frame Index')
        plt.ylabel('Mask Confidence (Max Logit)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    
    # Plot 4: Mask area over time grouped by object class
    if 'mask_area' in df.columns:
        plt.subplot(2, 2, 4)
        tracking_df = df_plot[df_plot['tracking_type'] == 'propagation']
        
        # Group by object class and combine data
        for obj_class in tracking_df['object_class'].unique():
            class_data = tracking_df[tracking_df['object_class'] == obj_class]
            # Sort by frame index to create smooth lines
            class_data = class_data.sort_values('frame_idx')
            plt.plot(class_data['frame_idx'], class_data['mask_area'], 
                    marker='.', alpha=0.6, label=f'{obj_class}')
            
        plt.title(f'Mask Area Over Time{plot_title_suffix}')
        plt.xlabel('Frame Index')
        plt.ylabel('Mask Area (pixels)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    confidence_plot_path = os.path.join(output_dir, "confidence_analysis.png")
    plt.savefig(confidence_plot_path, dpi=300, bbox_inches='tight')
    print(f"Confidence plot saved to: {confidence_plot_path}")
    
    # Print summary statistics
    print("\n=== Confidence Analysis Summary ===")
    print(f"Total confidence records: {len(df)}")
    print(f"Unique objects tracked: {df['object_id'].nunique()}")
    print(f"Frame range: {df['frame_idx'].min()} - {df['frame_idx'].max()}")
    
    # Show object classes detected
    if 'object_class' in df.columns:
        unique_classes = df['object_class'].unique()
        print(f"Object classes detected: {', '.join(unique_classes)}")
        
        # Show count by class
        class_counts = df.groupby('object_class')['object_id'].nunique().sort_values(ascending=False)
        print("\nObjects per class:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} objects")
    
    if 'detection_confidence' in df.columns:
        det_df = df[df['tracking_type'] == 'initial_detection']
        print(f"\nAverage detection confidence: {det_df['detection_confidence'].mean():.3f}")
        print(f"Detection confidence range: {det_df['detection_confidence'].min():.3f} - {det_df['detection_confidence'].max():.3f}")
        
        # Show confidence by class
        if 'object_class' in det_df.columns:
            print("\nDetection confidence by class:")
            class_conf = det_df.groupby('object_class')['detection_confidence'].agg(['mean', 'count']).round(3)
            for class_name, stats in class_conf.iterrows():
                print(f"  {class_name}: {stats['mean']:.3f} (n={stats['count']})")
    
    if 'mask_confidence' in df.columns:
        track_df = df[df['tracking_type'] == 'propagation']
        print(f"\nAverage tracking confidence: {track_df['mask_confidence'].mean():.3f}")
        print(f"Tracking confidence range: {track_df['mask_confidence'].min():.3f} - {track_df['mask_confidence'].max():.3f}")
    
    print("\nConfidence analysis completed!")
else:
    print("No confidence data collected.")
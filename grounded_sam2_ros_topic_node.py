#!/usr/bin/env python3
# ROS2 node: subscribe to an image topic and run detection or segmentation with Grounded-SAM2
# Min deps: rclpy, cv_bridge, numpy, torch, opencv-python, vision_msgs
# Models: groundingdino, sam2 (only needed for chosen mode)

import os
import sys
import time
from typing import List, Tuple, Optional

import numpy as np
import cv2
import rclpy
import torch

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge

# ---------- Optional model imports (handled gracefully) ----------
_HAS_DINO = False
_HAS_SAM2 = False

try:
    # GroundingDINO convenience wrapper (common in Grounded-SAM2 repos)
    # If your repo exposes a different API, adjust the import & .predict call below.
    from groundingdino.util.inference import Model as GroundingDINO
    _HAS_DINO = True
except Exception:
    pass

try:
    # Typical SAM2 imports; exact paths can vary by fork.
    # Adjust if your Grounded-SAM-2 repo uses different entry points.
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _HAS_SAM2 = True
except Exception:
    pass


def _to_detection_msg(header: Header, boxes_xyxy: np.ndarray, labels: List[str], scores: np.ndarray) -> Detection2DArray:
    """Convert detection results to vision_msgs/Detection2DArray."""
    msg = Detection2DArray()
    msg.header = header

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(float)
        w, h = (x2 - x1), (y2 - y1)

        det = Detection2D()
        det.header = header

        det.bbox = BoundingBox2D()
        det.bbox.center.x = x1 + w / 2.0
        det.bbox.center.y = y1 + h / 2.0
        det.bbox.size_x = w
        det.bbox.size_y = h

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = labels[i] if i < len(labels) else ""
        hyp.hypothesis.score = float(scores[i]) if i < len(scores) else 0.0
        det.results.append(hyp)

        msg.detections.append(det)
    return msg


def _overlay(
    img_bgr: np.ndarray,
    boxes_xyxy: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    scores: Optional[np.ndarray] = None,
    masks: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Overlay boxes/labels and optional masks on the image."""
    vis = img_bgr.copy()

    if masks is not None:
        for m in masks:
            if m is None:
                continue
            # mask expected as bool or 0/1, same size as image HxW
            ms = (m.astype(np.uint8) * 255)
            colored = cv2.applyColorMap(ms, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 1.0, colored, 0.4, 0)

    if boxes_xyxy is not None:
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = labels[i] if labels and i < len(labels) else "obj"
            score = scores[i] if scores is not None and i < len(scores) else None
            text = f"{label}" + (f" {score:.2f}" if score is not None else "")
            cv2.putText(vis, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


class GroundedSAMNode(Node):
    """
    Parameters:
      ~image_topic            (string)  default: /camera/image_color
      ~task                   (string)  "detection" or "segmentation"
      ~text_prompt            (string)  e.g., "boat, buoy, dock"
      ~box_threshold          (double)  e.g., 0.3
      ~text_threshold         (double)  e.g., 0.25
      ~dino_config            (string)  path to GroundingDINO cfg (if required by your wrapper)
      ~dino_checkpoint        (string)  path to GroundingDINO weights
      ~sam2_config            (string)  path to SAM2 config (if needed)
      ~sam2_checkpoint        (string)  path to SAM2 checkpoint
      ~publish_viz            (bool)    publish annotated image
      ~viz_topic              (string)  default: /groundedsam/viz
      ~mask_topic             (string)  default: /groundedsam/mask  (mono8)
      ~detections_topic       (string)  default: /groundedsam/detections
    """

    def __init__(self):
        super().__init__("groundedsam_node")

        # --- Parameters ---
        self.declare_parameter("image_topic", "/lucid/image_raw_rosclock")
        self.declare_parameter("task", "detection")  # "detection" | "segmentation"
        self.declare_parameter("text_prompt", "boat, dock, truck, water")
        self.declare_parameter("box_threshold", 0.3)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("dino_config", "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        self.declare_parameter("dino_checkpoint", "gdino_checkpoints/groundingdino_swint_ogc.pth")
        self.declare_parameter("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.declare_parameter("sam2_checkpoint", "./checkpoints/sam2.1_hiera_large.pt")
        self.declare_parameter("publish_viz", True)
        self.declare_parameter("viz_topic", "/groundedsam/viz")
        self.declare_parameter("mask_topic", "/groundedsam/mask")
        self.declare_parameter("detections_topic", "/groundedsam/detections")

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.task = self.get_parameter("task").get_parameter_value().string_value.lower()
        self.text_prompt = self.get_parameter("text_prompt").get_parameter_value().string_value
        self.box_thresh = float(self.get_parameter("box_threshold").get_parameter_value().double_value)
        self.text_thresh = float(self.get_parameter("text_threshold").get_parameter_value().double_value)
        self.dino_cfg = self.get_parameter("dino_config").get_parameter_value().string_value
        self.dino_ckpt = self.get_parameter("dino_checkpoint").get_parameter_value().string_value
        self.sam2_cfg = self.get_parameter("sam2_config").get_parameter_value().string_value
        self.sam2_ckpt = self.get_parameter("sam2_checkpoint").get_parameter_value().string_value
        self.publish_viz = bool(self.get_parameter("publish_viz").get_parameter_value().bool_value)
        self.viz_topic = self.get_parameter("viz_topic").get_parameter_value().string_value
        self.mask_topic = self.get_parameter("mask_topic").get_parameter_value().string_value
        self.dets_topic = self.get_parameter("detections_topic").get_parameter_value().string_value

        # --- QoS (sensor data) ---
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # --- Subs/Pubs ---
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.cb_image, qos)

        self.pub_dets = self.create_publisher(Detection2DArray, self.dets_topic, 10)
        self.pub_viz: Optional[rclpy.publisher.Publisher] = None
        self.pub_mask: Optional[rclpy.publisher.Publisher] = None
        if self.publish_viz:
            self.pub_viz = self.create_publisher(Image, self.viz_topic, 10)
        if self.task == "segmentation":
            self.pub_mask = self.create_publisher(Image, self.mask_topic, 10)

        # --- Models ---
        self.detector = None
        self.segmentor = None

        self._load_models()

        self.get_logger().info(
            f"[groundedsam_node] Ready. task='{self.task}', prompt='{self.text_prompt}', "
            f"image_topic='{self.image_topic}'"
        )

    # ------------------ Model Loading ------------------

    def _load_models(self):
        if self.task not in ("detection", "segmentation"):
            self.get_logger().warn(f"Unknown task '{self.task}', defaulting to 'detection'")
            self.task = "detection"

        # GroundingDINO (always needed)
        if not _HAS_DINO:
            self.get_logger().error(
                "GroundingDINO not found. Install it (e.g., `pip install groundingdino` or your repo's editable install)."
            )
        else:
            try:
                # Many Grounded-SAM repos expose a simple wrapper like this:
                # GroundingDINO(model_config_path, model_checkpoint_path, device='cuda'/'cpu')
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.detector = GroundingDINO(
                    model_config_path=self.dino_cfg if self.dino_cfg else None,
                    model_checkpoint_path=self.dino_ckpt if self.dino_ckpt else None,
                    device=device,
                )
                self.get_logger().info(f"Loaded GroundingDINO on {device}")
            except Exception as e:
                self.get_logger().error(f"Failed to init GroundingDINO: {e}")

        # SAM2 only for segmentation mode
        if self.task == "segmentation":
            if not _HAS_SAM2:
                self.get_logger().error("SAM2 not found. Install it per your Grounded-SAM2 repo instructions.")
            else:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    sam2_net = build_sam2(self.sam2_cfg if self.sam2_cfg else None,
                                          self.sam2_ckpt if self.sam2_ckpt else None)
                    sam2_net.to(device)
                    self.segmentor = SAM2ImagePredictor(sam2_net)
                    self._sam2_device = device
                    self.get_logger().info(f"Loaded SAM2 on {device}")
                except Exception as e:
                    self.get_logger().error(f"Failed to init SAM2: {e}")

    # ------------------ Inference ------------------

    def _run_detection(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Returns (boxes_xyxy [N,4], labels [N], scores [N])
        """
        if self.detector is None:
            return np.zeros((0, 4), dtype=np.float32), [], np.zeros((0,), dtype=np.float32)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Common API in GroundingDINO wrappers:
        # predict_with_caption(image, caption, box_threshold, text_threshold)
        try:
            det = self.detector.predict_with_caption(
                image=img_rgb,
                caption=self.text_prompt,
                box_threshold=self.box_thresh,
                text_threshold=self.text_thresh,
            )
            # Expecting dict-like with "boxes", "logits", "phrases" or similar
            boxes = np.array(det.get("boxes", []), dtype=np.float32)  # xyxy
            scores = np.array(det.get("logits", []), dtype=np.float32)
            labels = list(det.get("phrases", []))
            return boxes, labels, scores
        except Exception as e:
            self.get_logger().error(f"GroundingDINO predict failed: {e}")
            return np.zeros((0, 4), dtype=np.float32), [], np.zeros((0,), dtype=np.float32)

    def _run_segmentation(self, img_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> List[np.ndarray]:
        """
        For each detection box, run SAM2 to get a binary mask. Returns a list of HxW boolean masks.
        """
        if self.segmentor is None or boxes_xyxy.size == 0:
            return []

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.segmentor.set_image(img_rgb)

        masks: List[np.ndarray] = []
        for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
            try:
                # Box prompt into SAM2; API may differ across forks.
                # Typical: masks, scores, logits = predictor.predict(box=np.array([x1, y1, x2, y2]))
                m, s, _ = self.segmentor.predict(box=np.array([x1, y1, x2, y2]))
                # predictor returns a stack of candidate masks; take the best by score
                if isinstance(s, (list, tuple)):
                    best_idx = int(np.argmax(np.array(s)))
                    mask = m[best_idx]
                else:
                    mask = m[0] if isinstance(m, (list, tuple)) else m
                # Ensure boolean 2D mask
                mask_bool = (mask > 0.5) if mask.dtype != bool else mask
                masks.append(mask_bool.astype(bool))
            except Exception as e:
                self.get_logger().warn(f"SAM2 predict failed for a box: {e}")
                masks.append(None)
        return masks

    # ------------------ ROS Callback ------------------

    def cb_image(self, msg: Image):
        t0 = time.time()
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # 1) Detect
        boxes, labels, scores = self._run_detection(img_bgr)

        # 2) Optional segmentation
        masks = None
        if self.task == "segmentation" and boxes.size > 0:
            masks = self._run_segmentation(img_bgr, boxes)

        # 3) Publish detections
        dets_msg = _to_detection_msg(msg.header, boxes, labels, scores)
        self.pub_dets.publish(dets_msg)

        # 4) Publish mask (first/union) if requested
        if self.pub_mask is not None and masks:
            # Publish the union mask for convenience; edit to publish one per object if you prefer.
            union = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            for m in masks:
                if m is not None:
                    union[m.astype(bool)] = 255
            mask_msg = self.bridge.cv2_to_imgmsg(union, encoding="mono8")
            mask_msg.header = msg.header
            self.pub_mask.publish(mask_msg)

        # 5) Publish visualization
        if self.pub_viz is not None:
            vis = _overlay(img_bgr, boxes, labels, scores, masks)
            viz_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            viz_msg.header = msg.header
            self.pub_viz.publish(viz_msg)

        dt = (time.time() - t0) * 1000.0
        self.get_logger().debug(f"Inference done: {len(boxes)} boxes in {dt:.1f} ms")

# ------------------ main ------------------

def main(args=None):
    rclpy.init(args=args)
    node = GroundedSAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

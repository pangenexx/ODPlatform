import numpy as np
from typing import List, Dict, Any, Optional

from .engine import Detection


def detections_to_json(detections: List[Detection]) -> List[Dict[str, Any]]:
    return [
        {
            "class_id": det.class_id,
            "class_name": det.class_name,
            "confidence": det.confidence,
            "bbox": list(det.bbox)
        }
        for det in detections
    ]


def filter_detections_by_confidence(detections: List[Detection], 
                                   min_confidence: float) -> List[Detection]:
    return [det for det in detections if det.confidence >= min_confidence]


def filter_detections_by_class(detections: List[Detection], 
                              class_names: List[str]) -> List[Detection]:
    return [det for det in detections if det.class_name in class_names]


def group_detections_by_class(detections: List[Detection]) -> Dict[str, List[Detection]]:
    groups = {}
    for det in detections:
        if det.class_name not in groups:
            groups[det.class_name] = []
        groups[det.class_name].append(det)
    return groups


def calculate_iou(bbox1: tuple, bbox2: tuple) -> float:
    x1, y1, x2, y2 = bbox1
    x1b, y1b, x2b, y2b = bbox2
    
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def nms(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    if not detections:
        return []
    
    sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    kept = []
    
    while sorted_detections:
        best = sorted_detections.pop(0)
        kept.append(best)
        
        sorted_detections = [
            det for det in sorted_detections
            if calculate_iou(best.bbox, det.bbox) < iou_threshold
        ]
    
    return kept


def scale_bbox(bbox: tuple, original_shape: tuple, target_shape: tuple) -> tuple:
    original_h, original_w = original_shape
    target_h, target_w = target_shape
    
    x1, y1, x2, y2 = bbox
    
    scale_x = target_w / original_w
    scale_y = target_h / original_h
    
    return (
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y
    )


def crop_detection(image: np.ndarray, detection: Detection) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    
    x1_pixel = int(x1 * w)
    y1_pixel = int(y1 * h)
    x2_pixel = int(x2 * w)
    y2_pixel = int(y2 * h)
    
    return image[y1_pixel:y2_pixel, x1_pixel:x2_pixel]


def get_detection_center(detection: Detection) -> tuple:
    x1, y1, x2, y2 = detection.bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def count_detections_by_class(detections: List[Detection]) -> Dict[str, int]:
    counts = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return counts


def normalize_bbox(bbox: tuple, image_shape: tuple) -> tuple:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return (x1 / w, y1 / h, x2 / w, y2 / h)


def denormalize_bbox(bbox: tuple, image_shape: tuple) -> tuple:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return (x1 * w, y1 * h, x2 * w, y2 * h)

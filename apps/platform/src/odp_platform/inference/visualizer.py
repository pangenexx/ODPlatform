import cv2
import numpy as np
from typing import List

from .engine import Detection


def draw_detections(image: np.ndarray, detections: List[Detection], color_map: dict = None) -> np.ndarray:
    if color_map is None:
        color_map = {}
    
    result = image.copy()
    h, w = image.shape[:2]
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1_pixel = int(x1 * w)
        y1_pixel = int(y1 * h)
        x2_pixel = int(x2 * w)
        y2_pixel = int(y2 * h)
        
        color = color_map.get(det.class_id, (0, 255, 0))
        
        cv2.rectangle(result, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), color, 2)
        
        label = f"{det.class_name}: {det.confidence:.2f}"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_label = max(y1_pixel, label_size[1])
        
        cv2.rectangle(result, (x1_pixel, y_label - label_size[1]), 
                      (x1_pixel + label_size[0], y_label + base_line), color, cv2.FILLED)
        cv2.putText(result, label, (x1_pixel, y_label), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result


def draw_detection_grid(image: np.ndarray, detections: List[Detection], 
                        grid_size: int = 4) -> np.ndarray:
    result = image.copy()
    h, w = image.shape[:2]
    
    cell_w = w // grid_size
    cell_h = h // grid_size
    
    for i in range(grid_size):
        cv2.line(result, (i * cell_w, 0), (i * cell_w, h), (100, 100, 100), 1)
        cv2.line(result, (0, i * cell_h), (w, i * cell_h), (100, 100, 100), 1)
    
    return draw_detections(result, detections)


def create_result_overlay(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    overlay = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        h, w = image.shape[:2]
        x1_pixel = int(x1 * w)
        y1_pixel = int(y1 * h)
        x2_pixel = int(x2 * w)
        y2_pixel = int(y2 * h)
        
        mask = np.zeros_like(image)
        mask[y1_pixel:y2_pixel, x1_pixel:x2_pixel] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.7, mask, 0.3, 0)
    
    return overlay


def draw_confidence_histogram(detections: List[Detection], width: int = 400, height: int = 200) -> np.ndarray:
    histogram = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    confidences = [d.confidence for d in detections]
    if not confidences:
        return histogram
    
    bins = 10
    hist, _ = np.histogram(confidences, bins=bins, range=(0, 1))
    max_count = max(hist) if hist.max() > 0 else 1
    
    bar_width = width // bins
    for i in range(bins):
        bar_height = int((hist[i] / max_count) * (height - 40))
        x = i * bar_width
        y = height - 20 - bar_height
        cv2.rectangle(histogram, (x + 2, y), (x + bar_width - 2, height - 20), (255, 100, 100), -1)
        cv2.rectangle(histogram, (x + 2, y), (x + bar_width - 2, height - 20), (200, 50, 50), 1)
    
    cv2.putText(histogram, "Confidence Distribution", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return histogram

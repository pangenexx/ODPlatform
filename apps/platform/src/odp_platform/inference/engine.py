from dataclasses import dataclass
from typing import Optional, List
import time
import numpy as np

import torch

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR

logger = get_logger(base_path=LOGGING_DIR, log_type="infer", logger_name="odp-infer")


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class InferenceResult:
    detections: List[Detection]
    inference_ms: float
    input_shape: tuple[int, int]


class Detector:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, device: str = ""):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        
        self._model = None
        self._class_names = []
        
        self._load_model()
    
    def _load_model(self):
        try:
            from ultralytics import YOLO
            
            self._model = YOLO(self.model_path)
            if hasattr(self._model, 'names'):
                self._class_names = list(self._model.names.values())
            logger.info(f"模型加载成功: {self.model_path}")
            logger.info(f"类别数量: {len(self._class_names)}")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def detect(self, image: np.ndarray) -> InferenceResult:
        if self._model is None:
            raise RuntimeError("模型未加载")
        
        t0 = time.time()
        
        device = self.device
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        results = self._model(
            image,
            conf=self.conf,
            iou=self.iou,
            device=device,
            verbose=False
        )
        
        inference_ms = (time.time() - t0) * 1000
        
        detections = []
        h, w = image.shape[:2]
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self._class_names[class_id] if class_id < len(self._class_names) else str(class_id)
                    
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1/w, y1/h, x2/w, y2/h)
                    ))
        
        return InferenceResult(
            detections=detections,
            inference_ms=inference_ms,
            input_shape=(h, w)
        )
    
    def get_class_names(self) -> List[str]:
        return self._class_names
    
    def set_confidence(self, conf: float):
        self.conf = conf
    
    def set_iou(self, iou: float):
        self.iou = iou
    
    def warmup(self, image_size: tuple = (640, 640)):
        """GPU JIT 预热，消除首次推理的 CUDA kernel 编译延迟。
        
        注意：仅在 CUDA 设备上有效，CPU 推理跳过。
        
        Args:
            image_size: 预热用的虚拟图片尺寸
        """
        device = self.device
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cpu":
            logger.debug("CPU 推理，跳过 warmup")
            return
        
        logger.debug(f"开始 GPU warmup，设备: {device}")
        dummy = np.zeros((*image_size, 3), dtype=np.uint8)
        result = self.detect(dummy)
        
        if not result.detections:
            logger.debug("warmup 完成，无检测结果（预期行为）")
        elif len(result.detections) > 0:
            logger.warning(f"warmup 在纯黑图上检测到 {len(result.detections)} 个目标，请检查 conf 阈值")

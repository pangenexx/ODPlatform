import time
import numpy as np
from typing import Dict, List, Optional

from .engine import Detector
from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR

logger = get_logger(base_path=LOGGING_DIR, log_type="infer", logger_name="odp-infer")


class BenchmarkResult:
    def __init__(self):
        self.inference_times: List[float] = []
        self.mean_time: float = 0.0
        self.std_time: float = 0.0
        self.min_time: float = 0.0
        self.max_time: float = 0.0
        self.fps: float = 0.0
        self.detection_counts: List[int] = []
    
    def compute_stats(self):
        if not self.inference_times:
            return
        
        self.mean_time = np.mean(self.inference_times)
        self.std_time = np.std(self.inference_times)
        self.min_time = np.min(self.inference_times)
        self.max_time = np.max(self.inference_times)
        self.fps = 1000.0 / self.mean_time
    
    def to_dict(self) -> Dict:
        return {
            "mean_ms": round(self.mean_time, 2),
            "std_ms": round(self.std_time, 2),
            "min_ms": round(self.min_time, 2),
            "max_ms": round(self.max_time, 2),
            "fps": round(self.fps, 2),
            "samples": len(self.inference_times),
            "avg_detections": round(np.mean(self.detection_counts), 2) if self.detection_counts else 0
        }


def benchmark_detector(detector: Detector, test_images: List[np.ndarray], 
                      warmup_runs: int = 5, benchmark_runs: int = 50) -> BenchmarkResult:
    result = BenchmarkResult()
    
    logger.info(f"开始基准测试: 预热 {warmup_runs} 次, 测试 {benchmark_runs} 次")
    
    for i in range(warmup_runs):
        detector.detect(test_images[i % len(test_images)])
    
    for i in range(benchmark_runs):
        image = test_images[i % len(test_images)]
        inference_result = detector.detect(image)
        
        result.inference_times.append(inference_result.inference_ms)
        result.detection_counts.append(len(inference_result.detections))
        
        if (i + 1) % 10 == 0:
            logger.info(f"进度: {i + 1}/{benchmark_runs}")
    
    result.compute_stats()
    
    logger.info(f"基准测试完成: {result.to_dict()}")
    
    return result


def profile_detection_latency(detector: Detector, image: np.ndarray, 
                             iterations: int = 100) -> List[float]:
    latencies = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        detector.detect(image)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return latencies


def compare_detectors(detector_configs: List[Dict], test_images: List[np.ndarray]) -> List[Dict]:
    results = []
    
    for config in detector_configs:
        logger.info(f"测试模型: {config['name']}")
        
        detector = Detector(config['model_path'], conf=config.get('conf', 0.25))
        benchmark_result = benchmark_detector(detector, test_images)
        
        result_dict = {
            "name": config['name'],
            "model_path": config['model_path'],
            **benchmark_result.to_dict()
        }
        results.append(result_dict)
        
        logger.info(f"结果: {result_dict}")
    
    return sorted(results, key=lambda x: x['mean_ms'])


def analyze_detection_distribution(detector: Detector, images: List[np.ndarray]) -> Dict:
    class_counts = {}
    confidence_distribution = []
    
    for image in images:
        result = detector.detect(image)
        
        for det in result.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
            confidence_distribution.append(det.confidence)
    
    return {
        "total_detections": sum(class_counts.values()),
        "class_distribution": class_counts,
        "avg_confidence": round(np.mean(confidence_distribution), 3) if confidence_distribution else 0,
        "confidence_std": round(np.std(confidence_distribution), 3) if confidence_distribution else 0
    }

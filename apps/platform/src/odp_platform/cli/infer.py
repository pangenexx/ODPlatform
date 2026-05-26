import argparse
import json
from pathlib import Path
from typing import Optional

from odp_platform.inference.engine import Detector
from odp_platform.inference.visualizer import draw_detections
from odp_platform.inference.sources import ImageSource, VideoWriter
from odp_platform.inference.utils import detections_to_json
from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR

logger = get_logger(base_path=LOGGING_DIR, log_type="infer", logger_name="odp-infer")


def run_inference(args):
    model_path = args.model
    input_path = args.input
    output_path = args.output
    conf = args.conf
    iou = args.iou
    
    logger.info(f"加载模型: {model_path}")
    detector = Detector(model_path, conf=conf, iou=iou)
    
    logger.info(f"处理输入: {input_path}")
    source = ImageSource(input_path)
    
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    video_writer = None
    if source.cap is not None:
        fps = source.get_fps() or 30.0
        video_output = Path(output_path) / "output.mp4" if output_path else "output.mp4"
        video_writer = VideoWriter(str(video_output), fps=fps)
    
    frame_count = 0
    for image in source:
        result = detector.detect(image)
        annotated = draw_detections(image, result.detections)
        
        if video_writer:
            video_writer.write(annotated)
        else:
            if output_path:
                output_file = output_dir / f"detection_{frame_count:04d}.jpg"
                import cv2
                cv2.imwrite(str(output_file), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            
            detections_json = detections_to_json(result.detections)
            print(json.dumps({
                "frame": frame_count,
                "inference_ms": round(result.inference_ms, 2),
                "detections": detections_json
            }, indent=2, ensure_ascii=False))
        
        frame_count += 1
    
    source.release()
    if video_writer:
        video_writer.release()
    
    logger.info(f"推理完成，共处理 {frame_count} 帧")


def run_benchmark(args):
    from odp_platform.inference.benchmark import benchmark_detector
    import cv2
    
    model_path = args.model
    image_path = args.image
    iterations = args.iterations
    
    logger.info(f"加载模型: {model_path}")
    detector = Detector(model_path)
    
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    test_images = [image] * 10
    
    result = benchmark_detector(detector, test_images, benchmark_runs=iterations)
    
    print(json.dumps(result.to_dict(), indent=2))


def main():
    parser = argparse.ArgumentParser(prog="odp-infer", description="ODPlatform 推理工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    detect_parser = subparsers.add_parser("detect", help="执行目标检测")
    detect_parser.add_argument("--model", "-m", required=True, help="模型路径")
    detect_parser.add_argument("--input", "-i", required=True, help="输入图片/视频/目录/摄像头")
    detect_parser.add_argument("--output", "-o", help="输出目录")
    detect_parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    detect_parser.add_argument("--iou", type=float, default=0.45, help="IOU阈值")
    
    benchmark_parser = subparsers.add_parser("benchmark", help="基准测试")
    benchmark_parser.add_argument("--model", "-m", required=True, help="模型路径")
    benchmark_parser.add_argument("--image", "-i", required=True, help="测试图片")
    benchmark_parser.add_argument("--iterations", "-n", type=int, default=50, help="测试次数")
    
    args = parser.parse_args()
    
    if args.command == "detect":
        run_inference(args)
    elif args.command == "benchmark":
        run_benchmark(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""odp-infer CLI entry point — mirrors odp-train structure.

Usage:
  odp-infer --source 0                       # camera
  odp-infer --source demo.mp4                # video
  odp-infer --source test.jpg                # single image
  odp-infer --source ./images/               # image folder
  odp-infer --source 0 --show                # camera + display window
  odp-infer --source demo.mp4 --conf 0.5     # override confidence
  odp-infer --source 0 --no-viz              # disable beautify
  odp-infer --source demo.mp4 --no-save      # run only, no output
  odp-infer --pipeline-yaml my_pipe.yaml     # custom pipeline config
"""
from __future__ import annotations

import argparse
import logging
import sys

from odp_platform.inference import InferService
from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-infer",
        description="YOLO inference — frame_source capture + ultralytics inference + visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  odp-infer --source 0                       # camera
  odp-infer --source demo.mp4                # video
  odp-infer --source test.jpg                # single image
  odp-infer --source ./images/               # image folder
  odp-infer --source 0 --show                # camera + display window
  odp-infer --source 0 --threaded            # threaded pipeline
  odp-infer --source demo.mp4 --conf 0.5     # override confidence
  odp-infer --source 0 --no-viz              # disable beautify
  odp-infer --source demo.mp4 --no-save      # run only, no output
  odp-infer --pipeline-yaml my_pipe.yaml     # custom pipeline config
        """,
    )

    # config files
    parser.add_argument("--yaml", type=str, default=None,
                        help="D5 infer.yaml path (YOLO predict params; default: RUNTIME_CONFIGS_DIR/infer.yaml)")
    parser.add_argument("--pipeline-yaml", dest="pipeline_yaml", type=str, default=None,
                        help="frame_source + visualization pipeline config path")

    # inference params (override defaults)
    parser.add_argument("--source", type=str, help="input source: image/video/dir/camera id")
    parser.add_argument("--model", type=str, help="model path/name")
    parser.add_argument("--conf", type=float, help="confidence threshold")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, help="input image size")
    parser.add_argument("--device", type=str, help="inference device (0/cpu/0,1)")
    parser.add_argument("--max-det", dest="max_det", type=int, help="max detections per image")
    parser.add_argument("--classes", type=int, nargs="+", help="class IDs to detect")
    parser.add_argument("--experiment-name", dest="experiment_name", type=str,
                        help="experiment name (output goes to runs/infer/<experiment_name>/)")
    parser.add_argument("--show", dest="show", action="store_true", default=None,
                        help="display window (requires GUI)")
    parser.add_argument("--no-save", dest="save", action="store_false", default=None,
                        help="don't save results to disk")

    # service-level options
    parser.add_argument("--threaded", action="store_true",
                        help="threaded pipeline (camera/RTSP recommended)")
    parser.add_argument("--warmup", dest="warmup_frames", type=int, default=0,
                        help="threaded: discard first N frames for camera warmup")
    parser.add_argument("--window-name", dest="window_name", type=str, default="odp-infer",
                        help="display window title")
    parser.add_argument("--no-viz", dest="beautify", action="store_false", default=True,
                        help="disable beautify visualization")
    parser.add_argument("--no-info", dest="show_info", action="store_false", default=True,
                        help="disable HUD overlay (FPS/frame count etc.)")
    parser.add_argument("--no-rename-log", dest="rename_log", action="store_false", default=True,
                        help="don't rename log file to <experiment>.log")

    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="log level")

    return parser


def _setup_logging(log_level: str) -> None:
    get_logger(
        base_path=LOGGING_DIR,
        log_type="infer",
        log_level=getattr(logging, log_level),
        temp_log=False,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level)
    log = logging.getLogger("odp_platform.cli.infer")

    NON_CONFIG_KEYS = {
        "yaml", "pipeline_yaml", "beautify", "rename_log", "log_level",
        "threaded", "warmup_frames", "window_name", "show_info",
    }
    cli_args = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in NON_CONFIG_KEYS
    }

    log.info(f"starting odp-infer, CLI fields: {list(cli_args.keys())}")
    try:
        service = InferService()
        result = service.predict(
            yaml_path=args.yaml,
            pipeline_yaml=args.pipeline_yaml,
            cli_args=cli_args,
            beautify=args.beautify,
            rename_log=args.rename_log,
            threaded=args.threaded,
            warmup_frames=args.warmup_frames,
            window_name=args.window_name,
            show_info=args.show_info,
        )
    except KeyboardInterrupt:
        log.warning("user interrupt (Ctrl+C)")
        return 130
    except Exception as e:
        log.error(f"unexpected exception: {e}", exc_info=True)
        return 1

    if result.success:
        log.info(f"OK. time={result.infer_time:.2f}s, output={result.output_dir}")
        return 0
    else:
        log.error(f"inference failed: {result.error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
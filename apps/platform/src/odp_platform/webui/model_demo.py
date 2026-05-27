from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from odp_platform.webui.utils import list_model_files


def _refresh_models():
    models = list_model_files()
    return gr.update(
        choices=models,
        value=models[0] if models else None,
        interactive=True,
    )


def _load_model(model_path: str):
    if not model_path:
        return None, "未选择模型", gr.update(interactive=False)
    try:
        from odp_platform.inference.engine import Detector
    except ImportError as exc:
        return None, f"推理模块未就绪: {exc}", gr.update(interactive=False)

    try:
        detector = Detector(model_path)
    except Exception as exc:
        return None, f"加载失败: {exc}", gr.update(interactive=False)
    return detector, f"已加载: {Path(model_path).name}", gr.update(interactive=True)


def _run_inference(
    image: Any,
    conf: float,
    iou: float,
    detector: Any,
) -> tuple[Any | None, list[dict[str, Any]], str]:
    if image is None:
        return None, [], "未选择图片"
    if detector is None:
        return None, [], "未加载模型"

    try:
        import numpy as np

        from odp_platform.inference.visualizer import draw_detections
    except ImportError as exc:
        return None, [], f"推理依赖未就绪: {exc}"

    try:
        detector.conf = float(conf)
        detector.iou = float(iou)
        if isinstance(image, np.ndarray):
            image_np = image
        else:
            image_np = np.array(image)
        if image_np.size == 0:
            return None, [], "图片数据为空"
        result = detector.detect(image_np)
        vis = draw_detections(image_np, result.detections)
        rows = [
            {
                "class": detection.class_name,
                "conf": round(detection.confidence, 4),
                "bbox": list(detection.bbox),
            }
            for detection in result.detections
        ]
        status = f"检测数: {len(rows)} | {result.inference_ms:.1f} ms"
        return vis, rows, status
    except Exception as exc:
        return None, [], f"推理失败: {exc}"


def create_model_demo_ui() -> None:
    models = list_model_files()
    detector_state = gr.State(None)

    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        refresh_btn = gr.Button("刷新")
        model_dd = gr.Dropdown(
            label="模型",
            choices=models,
            value=models[0] if models else None,
            filterable=True,
            interactive=True,
        )
        load_btn = gr.Button("加载", variant="primary")
        status = gr.Textbox(
            label="状态",
            value="未加载",
            interactive=False,
            scale=2,
            max_lines=1,
            elem_classes=["odp-status"],
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        conf_slider = gr.Slider(0.01, 0.99, 0.25, step=0.01, label="Confidence")
        iou_slider = gr.Slider(0.01, 0.99, 0.45, step=0.01, label="IoU")
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        image_in = gr.Image(type="numpy", label="输入", sources=["upload"])
        image_out = gr.Image(type="numpy", label="结果", container=True)
    infer_btn = gr.Button("推理", interactive=False)
    json_out = gr.JSON(label="检测列表")

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])
    load_btn.click(
        fn=_load_model,
        inputs=[model_dd],
        outputs=[detector_state, status, infer_btn],
    )
    infer_btn.click(
        fn=_run_inference,
        inputs=[image_in, conf_slider, iou_slider, detector_state],
        outputs=[image_out, json_out, status],
    )

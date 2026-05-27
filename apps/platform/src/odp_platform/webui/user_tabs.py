from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr

from odp_platform.webui.utils import list_images, list_model_files

logger = logging.getLogger(__name__)

_detector_cache: dict[str, Any] = {}


def _model_choices() -> list[str]:
    return list_model_files()


def _refresh_models() -> gr.update:
    models = _model_choices()
    return gr.update(
        choices=models,
        value=models[0] if models else None,
    )


def _get_or_create_detector(model_path: str, conf: float, iou: float) -> Any | None:
    if not model_path:
        return None
    try:
        from odp_platform.inference.engine import Detector
    except ImportError as exc:
        logger.error("推理模块未就绪: %s", exc)
        return None
    detector = _detector_cache.get(model_path)
    if detector is None:
        logger.info("首次加载模型: %s", model_path)
        try:
            detector = Detector(model_path)
        except Exception as exc:
            logger.error("模型加载失败 %s: %s", model_path, exc)
            return None
        detector._model_path = model_path
        _detector_cache[model_path] = detector
    detector.conf = float(conf)
    detector.iou = float(iou)
    return detector


def _run_single_detection(
    image: Any,
    model_path: str,
    conf: float,
    iou: float,
    detector_state: Any,
) -> tuple[Any | None, list[dict[str, Any]], str, Any]:
    if image is None:
        return None, [], "未选择图片", detector_state
    if not model_path:
        return None, [], "未选择模型", detector_state

    try:
        import numpy as np

        from odp_platform.inference.visualizer import draw_detections
    except ImportError as exc:
        return None, [], f"推理依赖未就绪: {exc}", detector_state

    try:
        if isinstance(image, np.ndarray):
            image_np = image
        else:
            image_np = np.array(image)
        if image_np.size == 0:
            return None, [], "图片数据为空", detector_state
        if image_np.ndim not in {2, 3}:
            return None, [], f"图片维度异常: {image_np.ndim}", detector_state

        is_cached = (
            detector_state is not None
            and hasattr(detector_state, '_model_path')
            and detector_state._model_path == model_path
        )
        if is_cached:
            detector = detector_state
            detector.conf = float(conf)
            detector.iou = float(iou)
        else:
            from odp_platform.inference.engine import Detector
            detector = Detector(model_path)
            detector.conf = float(conf)
            detector.iou = float(iou)
            detector._model_path = model_path

        result = detector.detect(image_np)
        rendered = draw_detections(image_np, result.detections)
        rows = [
            {
                "class": detection.class_name,
                "conf": round(detection.confidence, 4),
                "bbox": list(detection.bbox),
            }
            for detection in result.detections
        ]
        status = f"{Path(model_path).name} | 检测数: {len(rows)} | {result.inference_ms:.1f} ms"
        return rendered, rows, status, detector
    except Exception as exc:
        logger.exception("单图检测失败")
        return None, [], f"检测失败: {exc}", detector_state


def _run_folder_detection(
    folder_path: str,
    model_path: str,
    conf: float,
    iou: float,
    detector_state: Any,
) -> tuple[list[Any], str, Any]:
    if not folder_path:
        return [], "请输入文件夹路径", detector_state
    if not model_path:
        return [], "未选择模型", detector_state

    try:
        import numpy as np
        from PIL import Image

        from odp_platform.inference.visualizer import draw_detections
    except ImportError as exc:
        return [], f"依赖未就绪: {exc}", detector_state

    folder = Path(folder_path)
    if not folder.is_dir():
        return [], f"路径不存在或不是文件夹: {folder_path}", detector_state

    image_paths = list_images(folder)
    if not image_paths:
        return [], f"文件夹中无图片: {folder_path}", detector_state

    is_cached = (
        detector_state is not None
        and hasattr(detector_state, '_model_path')
        and detector_state._model_path == model_path
    )
    if is_cached:
        detector = detector_state
    else:
        try:
            from odp_platform.inference.engine import Detector
            detector = Detector(model_path)
        except Exception as exc:
            return [], f"模型加载失败: {exc}", detector_state
        detector._model_path = model_path
    detector.conf = float(conf)
    detector.iou = float(iou)

    results = []
    for img_path in image_paths[:100]:
        try:
            image = np.array(Image.open(img_path))
            result = detector.detect(image)
            rendered = draw_detections(image, result.detections)
            results.append(rendered)
        except Exception as exc:
            logger.warning("跳过图片 %s: %s", img_path, exc)
            continue

    status = f"处理完成: {len(results)}/{len(image_paths)} 张 | 模型: {Path(model_path).name}"
    return results, status, detector


def _process_webcam_frame(
    frame: Any,
    model_path: str,
    conf: float,
    iou: float,
) -> Any:
    if frame is None or not model_path:
        return frame
    try:
        import numpy as np

        from odp_platform.inference.visualizer import draw_detections
    except ImportError:
        logger.error("webcam: 推理依赖未就绪")
        return frame

    try:
        if isinstance(frame, np.ndarray):
            image_np = frame
        else:
            image_np = np.array(frame)
        if image_np.size == 0 or image_np.ndim not in {2, 3}:
            return frame

        detector = _get_or_create_detector(model_path, conf, iou)
        if detector is None:
            return frame

        result = detector.detect(image_np)
        rendered = draw_detections(image_np, result.detections)
        return rendered
    except Exception as exc:
        logger.warning("webcam 流检测帧失败: %s", exc)
        return frame


def _select_model(model_path: str) -> str:
    if not model_path:
        return "未选择模型"
    return f"当前模型: {Path(model_path).name}"


def _chat(message: str, history: list[dict[str, str]] | None) -> tuple[list[dict[str, str]], str]:
    text = (message or "").strip()
    history = list(history or [])
    if not text:
        return history, ""

    history.append({"role": "user", "content": text})
    history.append(
        {
            "role": "assistant",
            "content": "LLM 接口待接入，前端入口已预留。",
        }
    )
    return history, ""


def _clear_chat() -> tuple[list[dict[str, str]], str]:
    return [], ""


def create_image_detection_ui() -> None:
    models = _model_choices()
    detector_state = gr.State(None)

    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        model_dd = gr.Dropdown(
            choices=models,
            value=models[0] if models else None,
            label="模型",
            filterable=True,
            interactive=True,
        )
        refresh_btn = gr.Button("刷新")
        conf_slider = gr.Slider(0.01, 0.99, 0.25, step=0.01, label="Confidence")
        iou_slider = gr.Slider(0.01, 0.99, 0.45, step=0.01, label="IoU")

    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        with gr.Column(scale=1):
            image_in = gr.Image(type="numpy", label="点击上传图片", sources=["upload"])
            detect_btn = gr.Button("开始检测", variant="primary")
        with gr.Column(scale=1):
            folder_in = gr.Textbox(
                label="图片文件夹路径",
                placeholder="eg. F:/datasets/test_images",
                max_lines=1,
            )
            folder_detect_btn = gr.Button("处理文件夹", variant="primary")

    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        image_out = gr.Image(type="numpy", label="检测结果", container=True)
        folder_gallery = gr.Gallery(label="文件夹检测结果", columns=3, height=400)

    status = gr.Textbox(label="状态", value="等待检测", interactive=False, max_lines=1)
    result_json = gr.JSON(label="检测列表")

    gr.Markdown("### 实时摄像头检测")
    webcam_status = gr.Textbox(
        label="摄像头状态",
        value="点击上方摄像头画面激活",
        max_lines=1,
        interactive=False,
    )
    gr.Markdown(
        "💡 **使用说明**：① 点击上方「摄像头」画面  ② 浏览器弹出权限请求时点击「允许」  "
        "③ 画面出现后自动开始逐帧推理。如果使用虚拟摄像头（VTubeStudioCam等），"
        "请先在系统中开启虚拟摄像头再刷新页面。"
    )
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        webcam_in = gr.Image(sources="webcam", type="numpy", label="摄像头")
        webcam_out = gr.Image(streaming=True, label="实时检测结果", container=True)

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])

    detect_btn.click(
        fn=_run_single_detection,
        inputs=[image_in, model_dd, conf_slider, iou_slider, detector_state],
        outputs=[image_out, result_json, status, detector_state],
    )

    folder_detect_btn.click(
        fn=_run_folder_detection,
        inputs=[folder_in, model_dd, conf_slider, iou_slider, detector_state],
        outputs=[folder_gallery, status, detector_state],
    )

    webcam_in.stream(
        fn=_process_webcam_frame,
        inputs=[webcam_in, model_dd, conf_slider, iou_slider],
        outputs=[webcam_out],
        time_limit=300,
        stream_every=0.3,
        concurrency_limit=5,
    )


def create_detection_results_ui() -> None:
    with gr.Row(elem_classes=["odp-row", "odp-row-action"]):
        refresh_btn = gr.Button("刷新")
        status = gr.Textbox(label="状态", value="暂无检测结果", interactive=False, max_lines=1)
    results = gr.Dataframe(
        label="检测历史",
        headers=["ID", "模型", "状态", "目标数", "时间"],
        value=[],
        interactive=False,
        wrap=True,
    )
    details = gr.JSON(label="结果详情", value=[])

    refresh_btn.click(
        fn=lambda: ("暂无检测结果", [], []),
        outputs=[status, results, details],
    )


def create_model_selection_ui() -> None:
    models = _model_choices()
    with gr.Row(elem_classes=["odp-row", "odp-row-action"]):
        refresh_btn = gr.Button("刷新")
        model_dd = gr.Dropdown(
            choices=models,
            value=models[0] if models else None,
            label="可用模型",
            filterable=True,
            interactive=True,
            scale=3,
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-action"]):
        select_btn = gr.Button("设为当前模型", variant="primary")
        status = gr.Textbox(
            label="状态",
            value=_select_model(models[0]) if models else "未发现模型",
            interactive=False,
            max_lines=1,
            scale=3,
        )

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])
    select_btn.click(fn=_select_model, inputs=[model_dd], outputs=[status])


def create_llm_chat_ui() -> None:
    with gr.Row(elem_classes=["odp-row"]):
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="sk-... 必填",
                scale=1,
            )
        with gr.Column(scale=1):
            api_base = gr.Textbox(
                label="API Base URL",
                value="https://api.deepseek.com/v1",
                scale=1,
            )
        with gr.Column(scale=1):
            model_name = gr.Dropdown(
                label="模型",
                choices=[
                    "deepseek-chat",
                    "deepseek-reasoner",
                    "gpt-4o",
                    "gpt-3.5-turbo",
                ],
                value="deepseek-chat",
                interactive=True,
                scale=1,
            )
    chatbot = gr.Chatbot(label="对话", height=400)
    message = gr.Textbox(label="输入", placeholder="输入问题，按Enter发送", max_lines=3)
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        send_btn = gr.Button("发送", variant="primary", scale=1)
        clear_btn = gr.Button("清空对话", scale=1)

    send_btn.click(fn=_chat, inputs=[message, chatbot], outputs=[chatbot, message])
    message.submit(fn=_chat, inputs=[message, chatbot], outputs=[chatbot, message])
    clear_btn.click(fn=_clear_chat, outputs=[chatbot, message])


def create_user_info_ui() -> None:
    with gr.Row(elem_classes=["odp-row", "odp-row-three"]):
        gr.Textbox(label="用户名", value="guest", interactive=False, max_lines=1)
        gr.Textbox(label="角色", value="user", interactive=False, max_lines=1)
        gr.Textbox(label="状态", value="未登录", interactive=False, max_lines=1)
    gr.JSON(
        label="检测概览",
        value={
            "检测任务": 0,
            "已完成": 0,
            "最近模型": "",
        },
    )

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from odp_platform.common.paths import ROOT_DIR
from odp_platform.webui.utils import list_dataset_names, platform_env

logger = logging.getLogger(__name__)


_train_process: subprocess.Popen | None = None
_train_lock = threading.Lock()


def _run_training_impl(
    dataset: str,
    dataset_path: str,
    experiment_name: str,
    model: str,
    epochs: int,
    batch: int,
    imgsz: int,
    lr0: float,
    device: str,
    workers: int,
    no_validate: str,
    dry_run: str,
) -> str:
    global _train_process

    dataset_actual = dataset_path.strip() or dataset
    if not dataset_actual:
        return "请选择数据集或填入数据集路径"

    name = experiment_name.strip() or f"webui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args = [
        sys.executable, "-m", "odp_platform.cli.train",
        "--dataset", dataset_actual,
        "--name", name,
        "--model", model.strip() or "yolo11n.pt",
        "--epochs", str(int(epochs)),
        "--batch", str(int(batch)),
        "--imgsz", str(int(imgsz)),
        "--lr0", str(float(lr0)),
        "--workers", str(int(workers)),
    ]
    if device.strip():
        args.extend(["--device", device.strip()])
    if no_validate == "是":
        args.append("--no-validate")
    if dry_run == "是":
        args.append("--dry-run")

    with _train_lock:
        if _train_process is not None:
            return "已有训练任务正在运行，请先停止"
        _train_process = subprocess.Popen(
            args,
            cwd=ROOT_DIR,
            env=platform_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    output_lines = []
    timeout = 30 if dry_run == "是" else 7200
    deadline = time.time() + timeout

    try:
        while time.time() < deadline:
            with _train_lock:
                if _train_process is None:
                    output_lines.append("\n[训练已停止]")
                    break
                ret = _train_process.poll()
            if ret is not None:
                stdout, stderr = _train_process.communicate()
                if stdout:
                    output_lines.append(stdout)
                if stderr:
                    output_lines.append(f"[stderr]\n{stderr}")
                output_lines.append(f"\n退出码: {ret}")
                break
            time.sleep(0.5)

        if time.time() >= deadline:
            output_lines.append(f"\n训练超时 ({timeout}s)")
    finally:
        with _train_lock:
            _train_process = None

    return "\n".join(output_lines)


def _stop_training() -> str:
    global _train_process
    with _train_lock:
        if _train_process is None:
            return "没有正在运行的训练任务"
        _train_process.terminate()
        try:
            _train_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _train_process.kill()
            _train_process.wait()
        _train_process = None
    return "训练已终止"


def _refresh_datasets():
    datasets = list_dataset_names()
    return gr.update(choices=datasets, value=datasets[0] if datasets else None, interactive=True)


def create_training_ui() -> None:
    datasets = list_dataset_names()
    with gr.Row(elem_classes=["odp-row", "odp-row-five"]):
        refresh_btn = gr.Button("刷新")
        dataset_dd = gr.Dropdown(
            label="数据集",
            choices=datasets,
            value=datasets[0] if datasets else None,
            filterable=True,
            interactive=True,
        )
        dataset_path = gr.Textbox(
            label="数据集路径（可替代下拉选择）",
            placeholder="eg. configs/datasets/rsod.yaml",
            max_lines=1,
        )
        experiment_name = gr.Textbox(
            label="实验名",
            placeholder="webui_rsod_001",
            max_lines=1,
        )
        model = gr.Textbox(
            label="模型",
            value="yolo11n.pt",
            max_lines=1,
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        epochs = gr.Number(label="Epochs", value=1, precision=0, minimum=1)
        batch = gr.Number(label="Batch", value=1, precision=0, minimum=1)
        imgsz = gr.Number(label="Image Size", value=640, precision=0, minimum=32)
        lr0 = gr.Number(label="LR0", value=0.01, minimum=0.000001)
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        device = gr.Textbox(
            label="Device（空=auto）",
            value="",
            max_lines=1,
        )
        workers = gr.Number(label="Workers", value=2, precision=0, minimum=0)
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        no_validate = gr.Dropdown(label="跳过质检", choices=["否", "是"], value="否", filterable=False)
        dry_run = gr.Dropdown(label="仅生成配置（不训练）", choices=["否", "是"], value="否", filterable=False)
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        run_btn = gr.Button("开始训练", variant="primary", size="lg")
        stop_btn = gr.Button("停止训练", variant="stop", size="lg")
    output = gr.Code(label="输出", language="shell", lines=18)

    refresh_btn.click(fn=_refresh_datasets, outputs=[dataset_dd])
    run_btn.click(
        fn=_run_training_impl,
        inputs=[
            dataset_dd,
            dataset_path,
            experiment_name,
            model,
            epochs,
            batch,
            imgsz,
            lr0,
            device,
            workers,
            no_validate,
            dry_run,
        ],
        outputs=[output],
    )
    stop_btn.click(
        fn=_stop_training,
        outputs=[output],
    )

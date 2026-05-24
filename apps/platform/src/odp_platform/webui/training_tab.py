from __future__ import annotations

from datetime import datetime

import gradio as gr

from odp_platform.webui.utils import list_dataset_names, run_python_module


def _run_training(
    dataset: str,
    experiment_name: str,
    model: str,
    epochs: int,
    batch: int,
    imgsz: int,
    lr0: float,
    device: str,
    workers: int,
    no_validate: bool,
    dry_run: bool,
) -> str:
    if not dataset:
        return "请选择数据集"

    name = experiment_name.strip() or f"webui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args = [
        "--dataset",
        dataset,
        "--name",
        name,
        "--model",
        model.strip() or "yolo11n.pt",
        "--epochs",
        str(int(epochs)),
        "--batch",
        str(int(batch)),
        "--imgsz",
        str(int(imgsz)),
        "--lr0",
        str(float(lr0)),
        "--workers",
        str(int(workers)),
    ]
    if device.strip():
        args.extend(["--device", device.strip()])
    if no_validate:
        args.append("--no-validate")
    if dry_run:
        args.append("--dry-run")

    result = run_python_module(
        "odp_platform.cli.train",
        args,
        timeout=7200 if not dry_run else 300,
    )
    return result.render()


def _refresh_datasets() -> gr.update:
    datasets = list_dataset_names()
    return gr.update(choices=datasets, value=datasets[0] if datasets else None)


def create_training_ui() -> None:
    datasets = list_dataset_names()
    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        refresh_btn = gr.Button("刷新")
        dataset_dd = gr.Dropdown(
            label="数据集",
            choices=datasets,
            value=datasets[0] if datasets else None,
            interactive=True,
        )
        experiment_name = gr.Textbox(
            label="实验名",
            placeholder="webui_rsod_001",
            max_lines=1,
            html_attributes={"wrap": "off"},
        )
        model = gr.Textbox(
            label="模型",
            value="yolo11n.pt",
            max_lines=1,
            html_attributes={"wrap": "off"},
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        epochs = gr.Number(label="Epochs", value=1, precision=0, minimum=1)
        batch = gr.Number(label="Batch", value=1, precision=0, minimum=1)
        imgsz = gr.Number(label="Image Size", value=640, precision=0, minimum=32)
        lr0 = gr.Number(label="LR0", value=0.01, minimum=0.000001)
    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        device = gr.Textbox(
            label="Device",
            value="",
            max_lines=1,
            html_attributes={"wrap": "off"},
        )
        workers = gr.Number(label="Workers", value=2, precision=0, minimum=0)
        no_validate = gr.Checkbox(label="跳过质检", value=False)
        dry_run = gr.Checkbox(label="Dry Run", value=True)
    run_btn = gr.Button("启动", variant="primary")
    output = gr.Code(label="输出", language="shell", lines=18)

    refresh_btn.click(fn=_refresh_datasets, outputs=[dataset_dd])
    run_btn.click(
        fn=_run_training,
        inputs=[
            dataset_dd,
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

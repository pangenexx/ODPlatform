from __future__ import annotations

import gradio as gr

from odp_platform.webui.utils import list_dataset_names, run_python_module


def _run_validation(
    dataset: str,
    dataset_path: str,
    task: str,
    verbose: str,
    no_report: str,
) -> str:
    dataset_actual = dataset_path.strip() or dataset
    if not dataset_actual:
        return "请选择数据集或填入数据集路径"
    args = ["--dataset", dataset_actual]
    if task != "auto":
        args.extend(["--task", task])
    if verbose == "是":
        args.append("--verbose")
    if no_report == "是":
        args.append("--no-report")
    return run_python_module("odp_platform.cli.validate_data", args, timeout=600).render()


def _refresh_datasets():
    datasets = list_dataset_names()
    return gr.update(choices=datasets, value=datasets[0] if datasets else None, interactive=True)


def create_validation_ui() -> None:
    datasets = list_dataset_names()
    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        refresh_btn = gr.Button("刷新")
        dataset_dd = gr.Dropdown(
            label="数据集",
            choices=datasets,
            value=datasets[0] if datasets else None,
            filterable=True,
            interactive=True,
        )
        dataset_path = gr.Textbox(
            label="数据集路径（可替代下拉）",
            placeholder="eg. configs/datasets/rsod.yaml",
            max_lines=1,
        )
        task_dd = gr.Dropdown(
            label="任务",
            choices=["auto", "detect", "segment"],
            value="auto",
            filterable=False,
            interactive=True,
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        verbose = gr.Dropdown(label="Verbose", choices=["否", "是"], value="否", filterable=False)
        no_report = gr.Dropdown(label="不写报告", choices=["否", "是"], value="否", filterable=False)
    run_btn = gr.Button("运行质检", variant="primary")
    output = gr.Code(label="输出", language="shell", lines=20)

    refresh_btn.click(fn=_refresh_datasets, outputs=[dataset_dd])
    run_btn.click(
        fn=_run_validation,
        inputs=[dataset_dd, dataset_path, task_dd, verbose, no_report],
        outputs=[output],
    )

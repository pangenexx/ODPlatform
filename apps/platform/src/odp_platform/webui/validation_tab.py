from __future__ import annotations

import gradio as gr

from odp_platform.webui.utils import list_dataset_names, run_python_module


def _run_validation(
    dataset: str,
    task: str,
    verbose: bool,
    no_report: bool,
) -> str:
    if not dataset:
        return "请选择数据集"
    args = ["--dataset", dataset]
    if task != "auto":
        args.extend(["--task", task])
    if verbose:
        args.append("--verbose")
    if no_report:
        args.append("--no-report")
    return run_python_module("odp_platform.cli.validate_data", args, timeout=600).render()


def _refresh_datasets() -> gr.update:
    datasets = list_dataset_names()
    return gr.update(choices=datasets, value=datasets[0] if datasets else None)


def create_validation_ui() -> None:
    datasets = list_dataset_names()
    with gr.Row(elem_classes=["odp-row", "odp-row-five"]):
        refresh_btn = gr.Button("刷新")
        dataset_dd = gr.Dropdown(
            label="数据集",
            choices=datasets,
            value=datasets[0] if datasets else None,
            interactive=True,
        )
        task_dd = gr.Dropdown(
            label="任务",
            choices=["auto", "detect", "segment"],
            value="auto",
            interactive=True,
        )
        verbose = gr.Checkbox(label="Verbose", value=False)
        no_report = gr.Checkbox(label="不写报告", value=False)
    run_btn = gr.Button("运行质检", variant="primary")
    output = gr.Code(label="输出", language="shell", lines=20)

    refresh_btn.click(fn=_refresh_datasets, outputs=[dataset_dd])
    run_btn.click(
        fn=_run_validation,
        inputs=[dataset_dd, task_dd, verbose, no_report],
        outputs=[output],
    )

from __future__ import annotations

from pathlib import Path

import gradio as gr

from odp_platform.common.paths import CONFIGS_DIR
from odp_platform.webui.utils import CONFIG_TASKS, list_config_files, run_python_module


def _default_output(task: str) -> str:
    return str(CONFIGS_DIR / f"{task}_config.yaml")


def _generate_config(task: str, output_path: str, force: bool) -> tuple[str, gr.update]:
    target = output_path.strip() or _default_output(task)
    args = ["generate", "--task", task, "--output", target]
    if force:
        args.append("--force")
    result = run_python_module("odp_platform.cli.config_cli", args, timeout=300)
    configs = list_config_files()
    return result.render(), gr.update(choices=configs, value=target if Path(target).exists() else None)


def _validate_config(config_path: str, task: str, preview: bool) -> str:
    if not config_path:
        return "请选择配置文件"
    args = ["validate", "--config", config_path, "--task", task]
    if preview:
        args.append("--preview")
    return run_python_module("odp_platform.cli.config_cli", args, timeout=300).render()


def _trace_config(config_path: str, task: str, field: str) -> str:
    if not config_path:
        return "请选择配置文件"
    args = ["trace", "--config", config_path, "--task", task]
    if field.strip():
        args.extend(["--field", field.strip()])
    return run_python_module("odp_platform.cli.config_cli", args, timeout=300).render()


def _task_changed(task: str) -> str:
    return _default_output(task)


def _refresh_configs() -> gr.update:
    configs = list_config_files()
    return gr.update(choices=configs, value=configs[0] if configs else None)


def create_config_ui() -> None:
    configs = list_config_files()
    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        task_dd = gr.Dropdown(
            label="任务",
            choices=CONFIG_TASKS,
            value="train",
            interactive=True,
        )
        output_path = gr.Textbox(
            label="输出路径",
            value=_default_output("train"),
            scale=2,
            max_lines=1,
            html_attributes={"wrap": "off"},
        )
        force = gr.Checkbox(label="覆盖", value=False)
        generate_btn = gr.Button("生成", variant="primary")
    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        refresh_btn = gr.Button("刷新")
        config_dd = gr.Dropdown(
            label="配置文件",
            choices=configs,
            value=configs[0] if configs else None,
            interactive=True,
            scale=2,
        )
        validate_task = gr.Dropdown(
            label="验证任务",
            choices=CONFIG_TASKS,
            value="train",
            interactive=True,
        )
        preview = gr.Checkbox(label="Preview", value=False)
    with gr.Row(elem_classes=["odp-row", "odp-row-three"]):
        validate_btn = gr.Button("验证", variant="primary")
        field = gr.Textbox(
            label="字段",
            placeholder="imgsz",
            max_lines=1,
            html_attributes={"wrap": "off"},
        )
        trace_btn = gr.Button("追溯")
    output = gr.Code(label="输出", language="shell", lines=20)

    task_dd.change(fn=_task_changed, inputs=[task_dd], outputs=[output_path])
    refresh_btn.click(fn=_refresh_configs, outputs=[config_dd])
    generate_btn.click(
        fn=_generate_config,
        inputs=[task_dd, output_path, force],
        outputs=[output, config_dd],
    )
    validate_btn.click(
        fn=_validate_config,
        inputs=[config_dd, validate_task, preview],
        outputs=[output],
    )
    trace_btn.click(
        fn=_trace_config,
        inputs=[config_dd, validate_task, field],
        outputs=[output],
    )

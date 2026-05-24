from __future__ import annotations

import gradio as gr

from odp_platform.common.paths import CHECKPOINTS_DIR, CONFIG_RUNS_DIR, VALIDATION_RUNS_DIR
from odp_platform.webui.utils import (
    fetch_backend_json,
    file_mtime,
    list_dataset_names,
    list_model_files,
    recent_files,
    relative_to_root,
)

SUMMARY_HEADERS = ["指标", "值"]
EXPERIMENT_HEADERS = ["ID", "实验", "数据集", "状态", "mAP50", "模型"]
LOG_HEADERS = ["类型", "文件", "更新时间"]


def _summary_rows() -> list[list[str]]:
    validation_reports = recent_files(VALIDATION_RUNS_DIR, "report.json", limit=999)
    config_snapshots = recent_files(CONFIG_RUNS_DIR, "config_snapshot.json", limit=999)
    return [
        ["数据集配置", str(len(list_dataset_names()))],
        ["模型权重", str(len(list_model_files()))],
        ["质检报告", str(len(validation_reports))],
        ["配置快照", str(len(config_snapshots))],
    ]


def _experiment_rows() -> list[list[str]]:
    payload = fetch_backend_json("/api/experiments", {"limit": 20})
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("data") or payload.get("experiments")
    else:
        items = payload
    if not isinstance(items, list):
        return []

    rows: list[list[str]] = []
    for item in items[:20]:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                str(item.get("id", "")),
                str(item.get("name", "")),
                str(item.get("dataset", "")),
                str(item.get("status", "")),
                str(item.get("best_map50", item.get("map50", ""))),
                str(item.get("model_path", "")),
            ]
        )
    return rows


def _log_rows() -> list[list[str]]:
    roots = [
        ("模型", CHECKPOINTS_DIR),
        ("质检", VALIDATION_RUNS_DIR),
        ("配置", CONFIG_RUNS_DIR),
    ]
    rows: list[list[str]] = []
    for label, root in roots:
        for path in recent_files(root, "*", limit=5):
            rows.append([label, relative_to_root(path), file_mtime(path)])
    rows.sort(key=lambda row: row[2], reverse=True)
    return rows[:12]


def _refresh_dashboard() -> tuple[list[list[str]], list[list[str]], list[list[str]], str]:
    experiments = _experiment_rows()
    status = "后端在线" if experiments else "后端未连接或暂无实验"
    return _summary_rows(), experiments, _log_rows(), status


def create_dashboard_ui() -> None:
    summary, experiments, logs, status = _refresh_dashboard()
    with gr.Row(elem_classes=["odp-row", "odp-row-action"]):
        refresh_btn = gr.Button("刷新", variant="primary")
        backend_status = gr.Textbox(
            label="后端状态",
            value=status,
            interactive=False,
            scale=3,
            max_lines=1,
            html_attributes={"wrap": "off"},
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        summary_df = gr.Dataframe(
            label="概览",
            headers=SUMMARY_HEADERS,
            value=summary,
            interactive=False,
            wrap=True,
        )
        experiments_df = gr.Dataframe(
            label="最近实验",
            headers=EXPERIMENT_HEADERS,
            value=experiments,
            interactive=False,
            wrap=True,
        )
    logs_df = gr.Dataframe(
        label="最近产物",
        headers=LOG_HEADERS,
        value=logs,
        interactive=False,
        wrap=True,
    )

    refresh_btn.click(
        fn=_refresh_dashboard,
        outputs=[summary_df, experiments_df, logs_df, backend_status],
    )

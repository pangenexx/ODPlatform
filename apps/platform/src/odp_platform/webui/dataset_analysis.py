from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import gradio as gr
from PIL import Image

from odp_platform.inference.data_visualizer import (
    count_class_distribution,
    generate_heatmap_from_labels,
    plot_class_bar_chart,
    plot_class_pie_chart,
)
from odp_platform.webui.utils import (
    dataset_yaml,
    list_dataset_names,
    read_yaml,
    resolve_split_dir,
)


def _get_class_info(dataset: str) -> Tuple[List[str], int]:
    """获取数据集类别信息"""
    if not dataset:
        return [], 0
    yaml_path = dataset_yaml(dataset)
    config = read_yaml(yaml_path)
    names = config.get("names", {})
    if isinstance(names, list):
        class_names = names
    elif isinstance(names, dict):
        class_names = [names.get(i, f"class_{i}") for i in range(len(names))]
    else:
        class_names = []
    return class_names, len(class_names)


def _get_label_dir(images_dir: Path) -> Path:
    """获取标签目录"""
    parent = images_dir.parent
    label_dir = parent / "labels"
    if label_dir.exists():
        return label_dir
    alt_label_dir = images_dir.parent.parent / "labels" / images_dir.name
    if alt_label_dir.exists():
        return alt_label_dir
    return label_dir


def _analyze_dataset(dataset: str, split: str) -> Tuple[Image.Image, Image.Image, Image.Image, str]:
    """分析数据集并生成可视化"""
    if not dataset:
        empty_img = Image.new('RGB', (400, 300), color=(240, 240, 240))
        return empty_img, empty_img, empty_img, "请选择数据集"
    
    class_names, num_classes = _get_class_info(dataset)
    if not class_names:
        empty_img = Image.new('RGB', (400, 300), color=(240, 240, 240))
        return empty_img, empty_img, empty_img, "无法获取类别信息"
    
    yaml_path = dataset_yaml(dataset)
    config = read_yaml(yaml_path)
    images_dir = resolve_split_dir(config, yaml_path, split)
    label_dir = _get_label_dir(images_dir)
    
    class_counts = count_class_distribution(label_dir, num_classes)
    total_boxes = sum(class_counts)
    
    pie_chart = plot_class_pie_chart(class_counts, class_names)
    bar_chart = plot_class_bar_chart(class_counts, class_names)
    heatmap = generate_heatmap_from_labels(label_dir, images_dir)
    
    status = f"数据集: {dataset} | 划分: {split} | 类别数: {num_classes} | 标注框总数: {total_boxes}"
    return pie_chart, bar_chart, heatmap, status


def _dataset_changed(dataset: str, split: str) -> Tuple[Image.Image, Image.Image, Image.Image, str]:
    """数据集或划分变化时更新可视化"""
    return _analyze_dataset(dataset, split)


def _refresh_datasets() -> gr.update:
    """刷新数据集列表"""
    datasets = list_dataset_names()
    return gr.update(choices=datasets, value=datasets[0] if datasets else None)


def create_dataset_analysis_ui() -> None:
    """创建数据集分析UI面板"""
    datasets = list_dataset_names()
    initial_dataset = datasets[0] if datasets else None
    
    initial_pie = Image.new('RGB', (400, 300), color=(240, 240, 240))
    initial_bar = Image.new('RGB', (400, 300), color=(240, 240, 240))
    initial_heatmap = Image.new('RGB', (400, 300), color=(240, 240, 240))
    initial_status = "请选择数据集"
    
    if initial_dataset:
        initial_pie, initial_bar, initial_heatmap, initial_status = _analyze_dataset(initial_dataset, "train")
    
    with gr.Row(elem_classes=["odp-row", "odp-row-three"]):
        refresh_btn = gr.Button("刷新")
        dataset_dd = gr.Dropdown(
            label="数据集",
            choices=datasets,
            value=initial_dataset,
            interactive=True,
        )
        split_dd = gr.Radio(
            label="划分",
            choices=["train", "val", "test"],
            value="train",
            interactive=True,
        )
    
    status = gr.Textbox(
        label="状态",
        value=initial_status,
        interactive=False,
        scale=2,
        max_lines=1,
        html_attributes={"wrap": "off"},
    )
    
    with gr.Row(elem_classes=["odp-row"]):
        pie_out = gr.Image(label="类别分布(饼图)", value=initial_pie, type="pil")
    
    with gr.Row(elem_classes=["odp-row"]):
        bar_out = gr.Image(label="类别分布(柱状图)", value=initial_bar, type="pil")
    
    heatmap_out = gr.Image(label="目标分布热力图", value=initial_heatmap, type="pil")
    
    dataset_dd.change(
        fn=_dataset_changed,
        inputs=[dataset_dd, split_dd],
        outputs=[pie_out, bar_out, heatmap_out, status],
    )
    split_dd.change(
        fn=_dataset_changed,
        inputs=[dataset_dd, split_dd],
        outputs=[pie_out, bar_out, heatmap_out, status],
    )
    refresh_btn.click(
        fn=_refresh_datasets,
        outputs=[dataset_dd],
    )
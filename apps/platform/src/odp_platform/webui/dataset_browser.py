from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from odp_platform.webui.utils import (
    dataset_yaml,
    label_path_for_image,
    list_dataset_names,
    list_images,
    normalize_names,
    read_yaml,
    relative_to_root,
    resolve_split_dir,
)

INFO_HEADERS = ["字段", "值"]
COLORS = [
    (40, 160, 240),
    (68, 184, 120),
    (239, 98, 98),
    (221, 169, 40),
    (144, 108, 228),
    (44, 178, 178),
]


def _dataset_info(dataset: str) -> list[list[str]]:
    if not dataset:
        return []
    yaml_path = dataset_yaml(dataset)
    config = read_yaml(yaml_path)
    names = normalize_names(config.get("names"))
    rows = [
        ["YAML", relative_to_root(yaml_path)],
        ["类别数", str(config.get("nc", len(names)))],
        ["类别", ", ".join(names.values())],
    ]
    for split in ("train", "val", "test"):
        images_dir = resolve_split_dir(config, yaml_path, split)
        rows.append([f"{split} 图片", str(len(list_images(images_dir)))])
    return rows


def _split_images(dataset: str, split: str) -> tuple[Path, list[Path], dict[str, Any]]:
    yaml_path = dataset_yaml(dataset)
    config = read_yaml(yaml_path)
    images_dir = resolve_split_dir(config, yaml_path, split)
    return images_dir, list_images(images_dir), config


def _parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        cx, cy, width, height = (float(value) for value in parts[1:5])
    except ValueError:
        return None
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)
    return cls_id, cx, cy, width, height


def draw_yolo_boxes(
    image: Image.Image,
    label_path: Path,
    names: dict[int, str],
) -> tuple[Image.Image, int]:
    vis = image.convert("RGB").copy()
    if not label_path.is_file():
        return vis, 0

    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()
    width, height = vis.size
    count = 0

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_yolo_line(line)
        if parsed is None:
            continue
        cls_id, cx, cy, box_w, box_h = parsed
        x1 = int((cx - box_w / 2) * width)
        y1 = int((cy - box_h / 2) * height)
        x2 = int((cx + box_w / 2) * width)
        y2 = int((cy + box_h / 2) * height)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width - 1), min(y2, height - 1)
        color = COLORS[cls_id % len(COLORS)]
        label = names.get(cls_id, str(cls_id))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        text_w = text_bbox[2] - text_bbox[0]
        y_text = max(0, y1 - text_h - 4)
        draw.rectangle((x1, y_text, x1 + text_w + 6, y_text + text_h + 4), fill=color)
        draw.text((x1 + 3, y_text + 2), label, fill=(255, 255, 255), font=font)
        count += 1
    return vis, count


def _preview(dataset: str, split: str, index: int) -> tuple[Image.Image | None, str]:
    if not dataset:
        return None, "未选择数据集"

    images_dir, images, config = _split_images(dataset, split)
    if not images:
        return None, f"{relative_to_root(images_dir)} 无图片"

    index = min(max(int(index), 0), len(images) - 1)
    image_path = images[index]
    label_path = label_path_for_image(image_path, images_dir)
    names = normalize_names(config.get("names"))
    image = Image.open(image_path)
    vis, count = draw_yolo_boxes(image, label_path, names)
    status = (
        f"{index + 1}/{len(images)} | {relative_to_root(image_path)} | "
        f"labels: {count}"
    )
    return vis, status


def _dataset_changed(dataset: str, split: str) -> tuple[
    list[list[str]],
    gr.update,
    Image.Image | None,
    str,
]:
    if not dataset:
        return [], gr.update(maximum=1, value=0, interactive=False), None, "无数据集"
    _, images, _ = _split_images(dataset, split)
    slider = gr.update(
        maximum=max(len(images) - 1, 1),
        value=0,
        interactive=bool(images),
    )
    image, status = _preview(dataset, split, 0)
    return _dataset_info(dataset), slider, image, status


def _refresh_datasets() -> tuple[gr.update, list[list[str]], gr.update, None, str]:
    datasets = list_dataset_names()
    value = datasets[0] if datasets else None
    info, slider, _, status = _dataset_changed(value, "train") if value else (
        [],
        gr.update(maximum=1, value=0, interactive=False),
        None,
        "无数据集",
    )
    return gr.update(choices=datasets, value=value), info, slider, None, status


def create_dataset_browser_ui() -> None:
    datasets = list_dataset_names()
    initial_dataset = datasets[0] if datasets else None
    initial_info, initial_slider, initial_image, initial_status = (
        _dataset_changed(initial_dataset, "train")
        if initial_dataset
        else ([], gr.update(maximum=1, value=0, interactive=False), None, "无数据集")
    )

    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
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
        index_slider = gr.Slider(
            label="图片",
            minimum=0,
            maximum=1,
            value=0,
            step=1,
            interactive=False,
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        info_df = gr.Dataframe(
            label="数据集信息",
            headers=INFO_HEADERS,
            value=initial_info,
            interactive=False,
            wrap=True,
        )
        status = gr.Textbox(
            label="当前样本",
            value=initial_status,
            interactive=False,
            scale=2,
            max_lines=1,
            html_attributes={"wrap": "off"},
        )
    image_out = gr.Image(label="预览", value=initial_image, type="pil")

    dataset_dd.change(
        fn=_dataset_changed,
        inputs=[dataset_dd, split_dd],
        outputs=[info_df, index_slider, image_out, status],
    )
    split_dd.change(
        fn=_dataset_changed,
        inputs=[dataset_dd, split_dd],
        outputs=[info_df, index_slider, image_out, status],
    )
    index_slider.change(
        fn=_preview,
        inputs=[dataset_dd, split_dd, index_slider],
        outputs=[image_out, status],
    )
    refresh_btn.click(
        fn=_refresh_datasets,
        outputs=[dataset_dd, info_df, index_slider, image_out, status],
    )

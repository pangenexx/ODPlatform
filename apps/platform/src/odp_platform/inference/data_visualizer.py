from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _setup_chinese_font():
    """设置matplotlib中文字体"""
    try:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


def count_class_distribution(label_dir: Path, num_classes: int) -> List[int]:
    """统计每个类别的样本数量（支持YOLO和VisDrone格式）"""
    counts = [0] * num_classes
    if not label_dir.exists():
        return counts

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()

                if len(parts) >= 5 and float(parts[1]) < 1.0:
                    try:
                        cls_id = int(float(parts[0]))
                        if 0 <= cls_id < num_classes:
                            counts[cls_id] += 1
                    except ValueError:
                        continue
                else:
                    parts = line.split(',')
                    if len(parts) >= 6:
                        try:
                            cls_id = int(float(parts[5]))
                            if 0 <= cls_id < num_classes:
                                counts[cls_id] += 1
                        except ValueError:
                            continue
    return counts


def plot_class_pie_chart(class_counts: List[int], class_names: List[str]) -> Image.Image:
    """绘制类别分布扇形统计图"""
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(14, 11))

    valid_indices = [i for i, count in enumerate(class_counts) if count > 0]
    valid_counts = [class_counts[i] for i in valid_indices]
    valid_names = [class_names[i] for i in valid_indices]

    if not valid_counts:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=24)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    total = sum(valid_counts)
    threshold = 0.05

    explode = [0.1 if c / total < threshold else 0 for c in valid_counts]
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_counts)))

    wedges, _ = ax.pie(
        valid_counts,
        labels=None,
        startangle=90,
        explode=explode,
        colors=colors,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
    )

    small_indices = []
    cumulative = 0.0
    for i, count in enumerate(valid_counts):
        pct = count / total * 100
        wedge_angle = (cumulative + count / total * 0.5) * 360 + 90
        cumulative += count / total
        if pct < threshold * 100:
            small_indices.append((i, wedge_angle))
        elif pct >= 1.0:
            if wedge_angle > 180:
                wedge_angle -= 360
            x = 0.65 * np.cos(np.radians(wedge_angle))
            y = 0.65 * np.sin(np.radians(wedge_angle))
            ax.annotate(f'{pct:.1f}%',
                       xy=(x, y),
                       ha='center', va='center',
                       fontsize=12, fontweight='bold', color='#1a1a1a')

    for i, wedge_angle in small_indices:
        count = valid_counts[i]
        pct = count / total * 100
        if wedge_angle > 180:
            wedge_angle -= 360
        inner_x = 0.95 * np.cos(np.radians(wedge_angle))
        inner_y = 0.95 * np.sin(np.radians(wedge_angle))
        outer_x = 1.18 * np.cos(np.radians(wedge_angle))
        outer_y = 1.18 * np.sin(np.radians(wedge_angle))
        ax.annotate(f'{pct:.1f}%',
                   xy=(outer_x, outer_y),
                   xytext=(outer_x, outer_y),
                   fontsize=11, fontweight='bold', color='#333333',
                   ha='center', va='center',
                   arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5,
                                   connectionstyle='arc3,rad=0'),
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f8f8', edgecolor='#aaaaaa'))

    legend_labels = [f'{name} ({count:,})' for name, count in zip(valid_names, valid_counts)]
    ax.legend(wedges, legend_labels, title="类别", loc="center left",
              bbox_to_anchor=(1.02, 0, 0.5, 1), fontsize=12, title_fontsize=14)

    ax.set_title('类别分布（饼图）', fontsize=18, fontweight='bold', pad=20)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def plot_class_bar_chart(class_counts: List[int], class_names: List[str]) -> Image.Image:
    """绘制类别分布柱状图"""
    _setup_chinese_font()
    plt.figure(figsize=(14, 8))

    valid_indices = [i for i, count in enumerate(class_counts) if count > 0]
    valid_counts = [class_counts[i] for i in valid_indices]
    valid_names = [class_names[i] for i in valid_indices]

    if not valid_counts:
        plt.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=20)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(valid_counts)))

    bars = plt.bar(range(len(valid_names)), valid_counts, color=colors, edgecolor='navy', linewidth=0.8)

    plt.xlabel('类别', fontsize=16, fontweight='bold')
    plt.ylabel('数量', fontsize=16, fontweight='bold')
    plt.title('类别分布（柱状图）', fontsize=18, fontweight='bold', pad=15)
    plt.xticks(range(len(valid_names)), valid_names, rotation=45, ha='right', fontsize=13)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, (bar, count) in enumerate(zip(bars, valid_counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + max(valid_counts) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=15, fontweight='bold')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def generate_heatmap_from_labels(label_dir: Path, image_dir: Path,
                                 grid_size: Tuple[int, int] = (10, 10)) -> Image.Image:
    """生成检测目标分布热力图（支持YOLO和VisDrone格式）"""
    _setup_chinese_font()
    heatmap = np.zeros(grid_size, dtype=np.float32)
    total_boxes = 0

    if not label_dir.exists() or not image_dir.exists():
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=20)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    img_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    img_width, img_height = 1920, 1080
    if img_files:
        try:
            with Image.open(img_files[0]) as img:
                img_width, img_height = img.size
        except Exception:
            pass

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    parts = line.split(',')

                if len(parts) >= 5:
                    try:
                        first_val = float(parts[0])
                        second_val = float(parts[1])

                        if second_val < 1.0 and len(parts) >= 5:
                            _, cx, cy, _, _ = map(float, parts[:5])
                        else:
                            bbox_left = float(parts[0])
                            bbox_top = float(parts[1])
                            bbox_width = float(parts[2])
                            bbox_height = float(parts[3])
                            cx = (bbox_left + bbox_width / 2) / img_width
                            cy = (bbox_top + bbox_height / 2) / img_height

                        gx = int(cx * grid_size[0])
                        gy = int(cy * grid_size[1])
                        gx = max(0, min(grid_size[0] - 1, gx))
                        gy = max(0, min(grid_size[1] - 1, gy))
                        heatmap[gy, gx] += 1
                        total_boxes += 1
                    except ValueError:
                        continue

    if total_boxes == 0:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "无标注数据", ha="center", va="center", fontsize=20)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    plt.figure(figsize=(12, 9))
    im = plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    cbar = plt.colorbar(im, label='目标密度', shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    plt.title('目标分布热力图', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('水平位置', fontsize=14)
    plt.ylabel('垂直位置', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def plot_map_curve(map_values: List[Tuple[float, float]], title: str = "mAP曲线") -> Image.Image:
    """绘制mAP曲线（mAP0.5:0.95）"""
    _setup_chinese_font()
    if not map_values:
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=20)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    iou_thresholds = [v[0] for v in map_values]
    map_scores = [v[1] for v in map_values]

    plt.figure(figsize=(12, 8))
    plt.plot(iou_thresholds, map_scores, marker='o', linestyle='-', color='#2563eb', linewidth=2.5, markersize=10)
    plt.xlabel('IoU阈值', fontsize=14)
    plt.ylabel('mAP', fontsize=14)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0.45, 1.0)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for iou, score in zip(iou_thresholds, map_scores):
        plt.text(iou, score + 0.02, f'{score:.3f}', ha='center', va='bottom', fontsize=11)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def plot_training_curve(epochs: List[int], train_loss: List[float],
                        val_loss: Optional[List[float]] = None,
                        train_map: Optional[List[float]] = None,
                        val_map: Optional[List[float]] = None) -> Image.Image:
    """绘制训练曲线"""
    _setup_chinese_font()
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='训练损失', color='#2563eb', linewidth=2)
    if val_loss:
        plt.plot(epochs, val_loss, label='验证损失', color='#dc2626', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('损失', fontsize=14)
    plt.title('损失曲线', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(1, 2, 2)
    if train_map:
        plt.plot(epochs, train_map, label='训练mAP', color='#2563eb', linewidth=2)
    if val_map:
        plt.plot(epochs, val_map, label='验证mAP', color='#dc2626', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('mAP', fontsize=14)
    plt.title('mAP曲线', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close()
    return Image.open(buf)
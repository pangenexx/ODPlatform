from __future__ import annotations

import csv
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from odp_platform.common.paths import CONFIGS_DATASETS_DIR, RUNS_DIR
from odp_platform.webui.utils import list_model_files

logger = logging.getLogger(__name__)


# =========================================================
# Tool Implementations
# =========================================================

def _tool_list_models() -> str:
    """列出所有可用模型。"""
    models = list_model_files()
    if not models:
        return "未找到任何 .pt 模型文件"
    lines = [f"共 {len(models)} 个模型:"] + [f"  - {m}" for m in models]
    return "\n".join(lines)


def _tool_list_datasets() -> str:
    """列出所有可用数据集。"""
    yamls = sorted(CONFIGS_DATASETS_DIR.glob("*.yaml"))
    if not yamls:
        return "未找到任何数据集配置文件"
    lines = [f"共 {len(yamls)} 个数据集:"]
    for y in yamls:
        lines.append(f"  - {y.stem}")
    return "\n".join(lines)


def _tool_list_experiments() -> str:
    """列出所有训练实验。"""
    exp_dir = RUNS_DIR / "experiments"
    if not exp_dir.exists():
        return "暂无训练实验"
    exps = sorted(
        [d.name for d in exp_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    if not exps:
        return "暂无训练实验"
    lines = [f"共 {len(exps)} 个实验:"]
    for name in exps:
        csv_path = exp_dir / name / "results.csv"
        if csv_path.exists():
            try:
                with open(csv_path) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    best_map50 = max(
                        float(r.get("metrics/mAP50(B)", 0)) for r in rows
                    )
                    lines.append(f"  - {name}  (best mAP50={best_map50:.4f})")
                    continue
            except Exception:
                pass
        lines.append(f"  - {name}")
    return "\n".join(lines)


def _tool_get_experiment(name: str) -> str:
    """获取指定实验的详细信息。"""
    exp_dir = RUNS_DIR / "experiments" / name
    if not exp_dir.exists():
        return f"实验不存在: {name}"

    csv_path = exp_dir / "results.csv"
    if not csv_path.exists():
        return f"实验 {name} 无 results.csv"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return f"实验 {name} 的 results.csv 为空"

    last = rows[-1]
    info = {
        "实验名": name,
        "总轮数": len(rows),
        "最后指标": {
            "mAP50": last.get("metrics/mAP50(B)", "N/A"),
            "mAP50-95": last.get("metrics/mAP50-95(B)", "N/A"),
            "precision": last.get("metrics/precision(B)", "N/A"),
            "recall": last.get("metrics/recall(B)", "N/A"),
        },
    }

    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
    info["最佳轮次"] = {
        "epoch": best.get("epoch", "N/A"),
        "mAP50": best.get("metrics/mAP50(B)", "N/A"),
    }

    config_path = exp_dir / "config_snapshot.json"
    if config_path.exists():
        try:
            info["配置"] = json.loads(config_path.read_text())
        except Exception:
            pass

    return json.dumps(info, ensure_ascii=False, indent=2)


def _tool_run_inference(
    model_path: str,
    image_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
) -> str:
    """对单张图片执行推理。"""
    model_file = Path(model_path)
    image_file = Path(image_path)

    if not model_file.exists():
        return f"模型文件不存在: {model_path}"
    if not image_file.exists():
        return f"图片文件不存在: {image_path}"

    try:
        from odp_platform.inference.engine import Detector
        from odp_platform.inference.visualizer import draw_detections
        import cv2
        import numpy as np

        detector = Detector(str(model_file))
        detector.warmup()

        img = cv2.imread(str(image_file))
        if img is None:
            return f"无法读取图片: {image_path}"
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = detector.detect(img_rgb)

        lines = [
            f"模型: {model_file.name}",
            f"图片: {image_file.name}",
            f"检测目标: {len(result.detections)} 个",
            f"推理耗时: {result.inference_ms:.1f}ms",
        ]
        for d in result.detections[:10]:
            lines.append(
                f"  - {d.class_name} "
                f"(置信度: {d.confidence:.3f}, "
                f"框: [{d.bbox[0]:.3f}, {d.bbox[1]:.3f}, "
                f"{d.bbox[2]:.3f}, {d.bbox[3]:.3f}])"
            )
        if len(result.detections) > 10:
            lines.append(f"  ... 还有 {len(result.detections) - 10} 个目标")
        return "\n".join(lines)

    except ImportError as e:
        return f"推理模块未就绪: {e}"
    except Exception as e:
        return f"推理失败: {e}"


def _tool_get_gpu_info() -> str:
    """获取 GPU 状态信息。"""
    try:
        import torch
    except ImportError:
        return "PyTorch 未安装"
    if not torch.cuda.is_available():
        return "GPU 不可用 (CUDA)"

    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free = total - reserved
        lines.append(
            f"GPU{i} [{props.name}]: "
            f"已用 {allocated:.1f}G / 空闲 {free:.1f}G / 总计 {total:.1f}G"
        )
    return "\n".join(lines)


# =========================================================
# Tool Registry
# =========================================================

_TOOLS: dict[str, tuple[dict[str, Any], Any]] = {}


def _register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    fn: Any,
) -> None:
    _TOOLS[name] = (
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": [k for k, v in parameters.items() if v.get("required")],
                },
            },
        },
        fn,
    )


_register_tool(
    "list_models",
    "列出所有可用的 .pt 模型文件",
    {},
    _tool_list_models,
)

_register_tool(
    "list_datasets",
    "列出所有可用的数据集（配置文件名称）",
    {},
    _tool_list_datasets,
)

_register_tool(
    "list_experiments",
    "列出所有训练实验及其最佳 mAP50",
    {},
    _tool_list_experiments,
)

_register_tool(
    "get_experiment",
    "获取指定训练实验的详细信息，包括指标和配置",
    {
        "name": {
            "type": "string",
            "description": "实验名称",
        },
    },
    _tool_get_experiment,
)

_register_tool(
    "run_inference",
    "对单张图片执行目标检测推理",
    {
        "model_path": {
            "type": "string",
            "description": "模型文件路径 (.pt)",
        },
        "image_path": {
            "type": "string",
            "description": "图片文件路径",
        },
        "conf": {
            "type": "number",
            "description": "置信度阈值 (默认 0.25)",
            "default": 0.25,
        },
        "iou": {
            "type": "number",
            "description": "IoU 阈值 (默认 0.45)",
            "default": 0.45,
        },
    },
    _tool_run_inference,
)

_register_tool(
    "get_gpu_info",
    "获取 GPU 显存使用状态",
    {},
    _tool_get_gpu_info,
)


# =========================================================
# Agent Loop
# =========================================================

_MODEL_TOOL_MAP = {
    "deepseek-v4-flash": None,
    "deepseek-v4-pro": None,
    "gpt-4o": None,
    "gpt-4o-mini": None,
}


def run_agent(
    messages: list[dict[str, str]],
    api_key: str,
    api_base: str,
    model_name: str,
) -> str:
    """执行 agent 循环：调用 LLM → 解析 function call → 执行工具 → 返回结果。

    Args:
        messages: 完整对话历史（含 system prompt）
        api_key: DeepSeek / OpenAI API Key
        api_base: API 基础 URL
        model_name: 模型名

    Returns:
        str: 最终回复内容
    """
    url = f"{api_base.rstrip('/')}/chat/completions"

    tool_schemas = [s for s, _ in _TOOLS.values()]

    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "tools": tool_schemas,
        "tool_choice": "auto",
        "stream": False,
        "max_tokens": 4096,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"API 请求失败 ({exc.code}): {body[:300]}"
    except Exception as exc:
        return f"请求异常: {exc}"

    choice = data["choices"][0]
    message = choice["message"]

    if not message.get("tool_calls"):
        return message.get("content", "")

    messages.append(message)

    for tool_call in message["tool_calls"]:
        func_name = tool_call["function"]["name"]
        try:
            func_args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            func_args = {}

        tool_fn = None
        for name, (_, fn) in _TOOLS.items():
            if name == func_name:
                tool_fn = fn
                break

        if tool_fn is None:
            result = f"未知工具: {func_name}"
        else:
            try:
                result = tool_fn(**func_args)
            except Exception as e:
                result = f"工具执行失败: {e}"
                logger.exception("工具执行异常: %s", func_name)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": str(result),
        })

    second_payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "stream": False,
        "max_tokens": 4096,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=second_payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"API 请求失败 ({exc.code}): {body[:300]}"
    except Exception as exc:
        return f"请求异常: {exc}"

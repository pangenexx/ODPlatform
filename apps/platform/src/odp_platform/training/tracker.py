from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import (
    DOCS_DIR,
    LOGGING_DIR,
    RUNS_DIR,
)

logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="train",
    logger_name="odp-train",
)


# =========================================================
# Data Structure
# =========================================================

@dataclass
class ExperimentSummary:
    """
    单个实验汇总信息
    """

    name: str
    dataset: str
    model: str

    epochs: int
    imgsz: int
    batch: int

    map50: float
    map50_95: float

    precision: float
    recall: float

    train_loss: float
    val_loss: float

    train_duration_sec: float

    experiment_dir: str
    model_path: str

    created_at: float

    # -----------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    # -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# Utils
# =========================================================

def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


# ---------------------------------------------------------

def _safe_int(v: Any) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


# ---------------------------------------------------------

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as e:
        logger.warning(f"JSON 读取失败: {path} {e}")
        return {}


# ---------------------------------------------------------

def _parse_results_csv(csv_path: Path) -> dict:
    """
    解析 ultralytics results.csv
    """

    if not csv_path.exists():
        logger.warning(f"results.csv 不存在: {csv_path}")
        return {}

    try:

        with open(csv_path, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {}

        last = rows[-1]

        metrics = {
            "epoch":
                _safe_int(last.get("epoch", 0)),

            "train_loss":
                _safe_float(last.get("train/box_loss", 0)),

            "val_loss":
                _safe_float(last.get("val/box_loss", 0)),

            "map50":
                _safe_float(last.get("metrics/mAP50(B)", 0)),

            "map50_95":
                _safe_float(last.get("metrics/mAP50-95(B)", 0)),

            "precision":
                _safe_float(last.get("metrics/precision(B)", 0)),

            "recall":
                _safe_float(last.get("metrics/recall(B)", 0)),

            "lr":
                _safe_float(last.get("lr/pg0", 0)),
        }

        return metrics

    except Exception as e:

        logger.warning(
            f"解析 results.csv 失败: {csv_path} {e}"
        )

        return {}


# ---------------------------------------------------------

def _load_experiment_summary(
    exp_dir: Path,
) -> ExperimentSummary | None:
    """
    从单个实验目录生成 summary
    """

    try:

        # =================================================
        # config
        # =================================================

        config_path = (
            exp_dir
            / "config_snapshot.json"
        )

        config = _read_json(config_path)

        if not config:
            logger.warning(
                f"缺少 config_snapshot.json: {exp_dir}"
            )
            return None

        # =================================================
        # metrics
        # =================================================

        results_csv = (
            exp_dir
            / "results.csv"
        )

        metrics = _parse_results_csv(results_csv)

        # =================================================
        # model
        # =================================================

        best_pt = (
            exp_dir
            / "weights"
            / "best.pt"
        )

        # =================================================
        # duration
        # =================================================

        train_duration = 0.0

        duration_path = (
            exp_dir
            / "train_duration.txt"
        )

        if duration_path.exists():

            try:

                train_duration = float(
                    duration_path.read_text().strip()
                )

            except Exception:
                pass

        # =================================================
        # summary
        # =================================================

        summary = ExperimentSummary(
            name=config.get(
                "name",
                exp_dir.name,
            ),

            dataset=config.get(
                "dataset",
                "unknown",
            ),

            model=config.get(
                "model",
                "",
            ),

            epochs=config.get(
                "epochs",
                0,
            ),

            imgsz=config.get(
                "imgsz",
                0,
            ),

            batch=config.get(
                "batch",
                0,
            ),

            map50=metrics.get(
                "map50",
                0,
            ),

            map50_95=metrics.get(
                "map50_95",
                0,
            ),

            precision=metrics.get(
                "precision",
                0,
            ),

            recall=metrics.get(
                "recall",
                0,
            ),

            train_loss=metrics.get(
                "train_loss",
                0,
            ),

            val_loss=metrics.get(
                "val_loss",
                0,
            ),

            train_duration_sec=train_duration,

            experiment_dir=str(
                exp_dir.resolve()
            ),

            model_path=str(
                best_pt.resolve()
            ),

            created_at=exp_dir.stat().st_mtime,
        )

        return summary

    except Exception as e:

        logger.exception(
            f"加载实验失败: {exp_dir}"
        )

        return None


# =========================================================
# Core
# =========================================================

def collect_results(
    dataset: str | None = None,
) -> list[ExperimentSummary]:
    """
    扫描 experiments/ 目录

    生成所有实验 summary
    """

    experiments_root = (
        RUNS_DIR
        / "experiments"
    )

    if not experiments_root.exists():

        logger.warning(
            f"实验目录不存在: {experiments_root}"
        )

        return []

    summaries: list[ExperimentSummary] = []

    exp_dirs = sorted(
        [
            p for p in experiments_root.iterdir()
            if p.is_dir()
        ]
    )

    logger.info(
        f"扫描 experiments: total={len(exp_dirs)}"
    )

    for exp_dir in exp_dirs:

        summary = _load_experiment_summary(
            exp_dir
        )

        if summary is None:
            continue

        # dataset filter
        if (
            dataset is not None
            and summary.dataset != dataset
        ):
            continue

        summaries.append(summary)

    # map50 降序
    summaries.sort(
        key=lambda x: x.map50,
        reverse=True,
    )

    logger.info(
        f"收集实验结果完成: total={len(summaries)}"
    )

    return summaries


# =========================================================
# CSV Export
# =========================================================

def export_comparison_csv(
    summaries: list[ExperimentSummary],
    output_path: Path,
) -> Path:
    """
    导出对比 CSV
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    headers = [
        "name",
        "dataset",
        "model",

        "epochs",
        "imgsz",
        "batch",

        "map50",
        "map50_95",

        "precision",
        "recall",

        "train_loss",
        "val_loss",

        "train_duration_sec",

        "model_path",
        "experiment_dir",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=headers,
        )

        writer.writeheader()

        for summary in summaries:

            writer.writerow({
                "name":
                    summary.name,

                "dataset":
                    summary.dataset,

                "model":
                    summary.model,

                "epochs":
                    summary.epochs,

                "imgsz":
                    summary.imgsz,

                "batch":
                    summary.batch,

                "map50":
                    f"{summary.map50:.6f}",

                "map50_95":
                    f"{summary.map50_95:.6f}",

                "precision":
                    f"{summary.precision:.6f}",

                "recall":
                    f"{summary.recall:.6f}",

                "train_loss":
                    f"{summary.train_loss:.6f}",

                "val_loss":
                    f"{summary.val_loss:.6f}",

                "train_duration_sec":
                    f"{summary.train_duration_sec:.2f}",

                "model_path":
                    summary.model_path,

                "experiment_dir":
                    summary.experiment_dir,
            })

    logger.info(
        f"comparison csv 已生成: {output_path}"
    )

    return output_path


# =========================================================
# High-level API
# =========================================================

def generate_comparison_report(
    dataset: str | None = None,
) -> Path:
    """
    生成实验对比 CSV

    输出：
        docs/results/comparison_*.csv
    """

    summaries = collect_results(dataset)

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    dataset_name = (
        dataset
        if dataset is not None
        else "all"
    )

    output_path = (
        DOCS_DIR
        / "results"
        / f"comparison_{dataset_name}_{timestamp}.csv"
    )

    export_comparison_csv(
        summaries,
        output_path,
    )

    logger.info(
        f"实验对比报告生成完成: {output_path}"
    )

    return output_path


# =========================================================
# Ranking
# =========================================================

def topk_experiments(
    k: int = 5,
    dataset: str | None = None,
) -> list[ExperimentSummary]:
    """
    获取 TopK 实验
    """

    summaries = collect_results(dataset)

    return summaries[:k]


# =========================================================
# CLI Debug
# =========================================================

if __name__ == "__main__":

    report_path = generate_comparison_report(
        dataset=None
    )

    print(f"\nCSV 已生成:\n{report_path}\n")

    top5 = topk_experiments(5)

    print("=" * 80)
    print("TOP5 EXPERIMENTS")
    print("=" * 80)

    for i, exp in enumerate(top5):

        print(
            f"[{i+1}] "
            f"{exp.name:<30} "
            f"mAP50={exp.map50:.4f} "
            f"P={exp.precision:.4f} "
            f"R={exp.recall:.4f}"
        )
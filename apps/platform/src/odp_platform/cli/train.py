from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from odp_platform.cli.validate_data import main as validate_main
from odp_platform.common.constants import AnnotationFormat, RunTask, Task
from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import (
    APP_DIR,
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    CONFIGS_DATASETS_DIR,
    DATA_DIR,
    LOGGING_DIR,
    RUNS_DIR,
    dataset_yaml_path,
)
from odp_platform.run_config import (
    ConfigSnapshot,
    build_config,
    generate_template_to_file,
    save_snapshot_to_file,
)
from odp_platform.training.callbacks import TrainingHooks

logger = logging.getLogger("odp-train")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-train",
        description=(
            "ODPlatform 集成训练命令 —— 一键完成：\n"
            "  1) 数据转换（如需）\n"
            "  2) 数据质检\n"
            "  3) 运行配置（含快照自动保存）\n"
            "  4) Ultralytics 模型训练\n"
            "所有参数、配置、来源追溯自动写入日志和快照。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset", "-d", required=True,
                        help="数据集名（= configs/datasets/<name>.yaml）")
    parser.add_argument("--task", "-t", default=Task.DETECT,
                        choices=Task.all(),
                        help="算法任务类型 (默认 detect)")

    parser.add_argument("--format", "-f",
                        choices=AnnotationFormat.all(),
                        help="原始标注格式（指定则先执行数据转换）")
    parser.add_argument("--train-rate", type=float, default=0.8)
    parser.add_argument("--val-rate", type=float, default=0.1)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="类别白名单（YOLO 格式必传）")
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--config", "-c", default=None,
                        help="训练配置 YAML（不指定则自动生成）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（覆盖配置中的值）")
    parser.add_argument("--batch", type=int, default=None,
                        help="批次大小（覆盖配置中的值）")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入图像尺寸（覆盖配置中的值）")
    parser.add_argument("--lr0", type=float, default=None,
                        help="初始学习率（覆盖配置中的值）")
    parser.add_argument("--model", default=None,
                        help="模型名称或权重路径（覆盖配置中的值）")
    parser.add_argument("--device", default=None,
                        help="训练设备（覆盖配置中的值）")
    parser.add_argument("--workers", type=int, default=None,
                        help="数据加载线程数（覆盖配置中的值）")
    parser.add_argument("--amp", type=bool, default=None,
                        help="自动混合精度（覆盖配置中的值）")
    parser.add_argument("--no-validate", action="store_true",
                        help="跳过数据质检步骤")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅生成配置和快照，不实际训练")
    parser.add_argument("--name", default=None,
                        help="实验名称（默认自动生成）")

    return parser


def _collect_cli_overrides(args) -> dict:
    overrides = {}
    for key in ("epochs", "batch", "imgsz", "lr0", "model", "device", "workers"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    if args.amp is not None:
        overrides["amp"] = args.amp
    return overrides


def _generate_experiment_name(model_arg: str | None) -> str:
    model_stem = "yolo"
    if model_arg:
        stem = Path(model_arg).stem
        if stem:
            model_stem = stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    experiments_dir = RUNS_DIR / "experiments"
    if experiments_dir.exists():
        existing = list(experiments_dir.glob("train-*"))
        counter = len(existing) + 1
    else:
        counter = 1
    return f"train-{counter}_{timestamp}_{model_stem}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    get_logger(base_path=LOGGING_DIR, log_type="train", log_level=logging.INFO,
               logger_name="odp-train")

    experiment_name = args.name or _generate_experiment_name(args.model)
    dataset = args.dataset
    task_type = args.task

    logger.info("=" * 60)
    logger.info(f"ODPlatform 集成训练 — {experiment_name}")
    logger.info(f"数据集: {dataset}  任务: {task_type}")
    logger.info("=" * 60)

    step_results = {}

    # ── Step 1: 数据转换 ──
    if args.format:
        logger.info(f"[1/4] 数据转换: {dataset} ({args.format} → YOLO)")
        try:
            from odp_platform.data_pipeline import DatasetPipeline
            pipeline = DatasetPipeline(
                dataset_name=dataset,
                annotation_format=args.format,
                task=task_type,
                train_rate=args.train_rate,
                val_rate=args.val_rate,
                random_state=args.random_state,
                classes=args.classes,
            )
            result = pipeline.run()
            logger.info(f"      转换完成: {result['yaml']}")
            step_results["transform"] = "success"
        except Exception as e:
            logger.error(f"      数据转换失败: {e}")
            return 2
    else:
        logger.info("[1/4] 跳过数据转换（未指定 --format）")

    # ── Step 2: 数据质检 ──
    if not args.no_validate:
        logger.info("[2/4] 数据质检")
        yaml_path = dataset_yaml_path(dataset)
        if not yaml_path.exists():
            logger.error(f"      数据集 YAML 不存在: {yaml_path}")
            return 2
        validate_argv = ["--dataset", dataset, "--task", task_type]
        exit_code = validate_main(validate_argv)
        if exit_code != 0:
            logger.warning(f"      质检完成，退出码: {exit_code}（继续训练）")
        else:
            logger.info("      质检通过 (exit 0)")
        step_results["validate"] = exit_code
    else:
        logger.info("[2/4] 跳过数据质检")

    # ── Step 3: 运行配置 ──
    logger.info("[3/4] 运行配置")

    config_path = args.config
    cli_overrides = _collect_cli_overrides(args)

    if config_path:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = APP_DIR / config_file
        logger.info(f"      加载配置: {config_file}")
        bundle = build_config(
            task="train",
            yaml_path=config_file,
            cli_args=cli_overrides if cli_overrides else None,
        )
    else:
        logger.info("      自动生成训练配置")
        auto_config = CONFIGS_DIR / f"{experiment_name}_config.yaml"
        generate_template_to_file(task="train", output_path=auto_config, force=True)
        bundle = build_config(
            task="train",
            yaml_path=auto_config,
            cli_args=cli_overrides if cli_overrides else None,
        )
        config_path = str(auto_config)

    if not bundle.valid:
        logger.error("      配置验证失败:")
        for err in bundle.errors:
            logger.error(f"        [ERROR] {err.field}: {err.message}")
        return 2

    snapshot = ConfigSnapshot.from_bundle(bundle)
    snapshot_run_dir = RUNS_DIR / "run_config" / experiment_name
    snapshot_run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_run_dir / "config_snapshot.json"
    save_snapshot_to_file(snapshot, snapshot_path)

    report_dict = bundle.to_report_dict(mask_sensitive=True)
    report_path = snapshot_run_dir / "config_report.json"
    report_path.write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"      配置快照: {snapshot_path}")
    logger.info(f"      配置报告: {report_path}")

    ultralytics_args = bundle.to_ultralytics_args()
    ultralytics_args["task"] = task_type
    logger.info(f"      Ultralytics 参数: {json.dumps(ultralytics_args, ensure_ascii=False)}")

    step_results["config"] = {
        "snapshot": str(snapshot_path),
        "report": str(report_path),
        "params": ultralytics_args,
    }

    # ── Step 4: 训练 ──
    if args.dry_run:
        logger.info("[4/4] 跳过训练（--dry-run）")
        logger.info(f"\n实验 '{experiment_name}' 的完整配置已保存:")
        logger.info(f"  配置快照: {snapshot_path}")
        logger.info(f"  配置报告: {report_path}")
        logger.info("\n直接训练命令:")
        cmd_parts = [
            "yolo",
            f"task={task_type}",
            f"mode=train",
            f"data={dataset_yaml_path(dataset)}",
        ]
        for k, v in ultralytics_args.items():
            cmd_parts.append(f"{k}={v}")
        logger.info(" ".join(cmd_parts))
        return 0

    logger.info(f"[4/4] 启动训练 — 模型: {ultralytics_args.get('model', 'yolo11n.pt')}")
    data_yaml = dataset_yaml_path(dataset)
    if not data_yaml.exists():
        logger.error(f"      数据集 YAML 不存在: {data_yaml}")
        return 2

    ultralytics_args["project"] = str(RUNS_DIR / "experiments")
    ultralytics_args["name"] = experiment_name
    exp_dir = RUNS_DIR / "experiments" / experiment_name

    hooks = TrainingHooks()
    config_json = json.dumps(ultralytics_args, ensure_ascii=False)
    hooks.on_train_start(
        name=experiment_name,
        config_json=config_json,
        dataset=dataset,
        model=ultralytics_args.get("model", "yolo11n.pt"),
    )

    try:
        model = YOLO(ultralytics_args.get("model", "yolo11n.pt"))
        results = model.train(
            data=str(data_yaml),
            **ultralytics_args,
        )
        logger.info(f"      训练完成")
        step_results["train"] = "success"

        results_csv = exp_dir / "results.csv"
        epoch_count = 0
        if results_csv.exists():
            with open(results_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    hooks.on_epoch_end_from_csv_row(row)
                    epoch_count += 1
            logger.info(f"      已同步 {epoch_count} 个 epoch 指标到数据库")

        best_pt = exp_dir / "weights" / "best.pt"
        checkpoint_path = ""
        if best_pt.exists():
            checkpoint_name = f"best_{experiment_name}.pt"
            checkpoint_path = str(CHECKPOINTS_DIR / checkpoint_name)
            shutil.copy(best_pt, checkpoint_path)
            logger.info(f"      模型权重已复制到 {checkpoint_path}")

        best_map50 = 0.0
        if results_csv.exists():
            with open(results_csv, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                best_map50 = max(float(r.get("metrics/mAP50(B)", 0)) for r in rows)

        hooks.on_train_end(best_map50=best_map50, model_path=checkpoint_path)
        return 0
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
        hooks.on_train_failed(reason="用户中断")
        return 3
    except Exception as e:
        logger.exception(f"训练失败: {e}")
        hooks.on_train_failed(reason=str(e))
        return 2
    finally:
        summary_path = snapshot_run_dir / "train_summary.json"
        summary = {
            "experiment": experiment_name,
            "dataset": dataset,
            "task": task_type,
            "timestamp": datetime.now().isoformat(),
            "steps": step_results,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"训练摘要: {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
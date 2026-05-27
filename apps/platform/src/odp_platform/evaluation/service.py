from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import (
    APP_DIR,
    CONFIGS_DIR,
    CONFIGS_DATASETS_DIR,
    LOGGING_DIR,
    RUNS_DIR,
)
from odp_platform.run_config import (
    ConfigSnapshot,
    build_config,
    generate_template_to_file,
    save_snapshot_to_file,
)

logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="val",
    logger_name="odp-val",
)


class ValResult:
    """验证结果，不可变数据容器。

    Attributes:
        success: 验证是否成功
        experiment_name: 实验名称
        output_dir: 输出目录 (runs/val/<experiment_name>/)
        metrics: 验证指标字典
        error: 错误信息（success=False 时有效）
        snapshot_path: 配置快照路径
    """

    def __init__(
        self,
        success: bool,
        experiment_name: str = "",
        output_dir: Optional[Path] = None,
        metrics: Optional[dict] = None,
        error: Optional[str] = None,
        snapshot_path: Optional[Path] = None,
    ):
        self.success = success
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.metrics = metrics or {}
        self.error = error
        self.snapshot_path = snapshot_path


class ValService:
    """D7: 模型评估编排器。

    设计原则（继承 D6）：
      - service 不写 addHandler / setLevel（CLI 入口通过 get_logger 完成）
      - 配置加载走 D5 run_config 子系统
      - 不包含跨任务通用工具（放在 common/）
      - service 永不抛：错误装进 ValResult.error

    用法:
        service = ValService()
        result = service.validate(
            model_path="best.pt",
            dataset="rsod",
            yaml_path="val.yaml",
        )
        if not result.success:
            print(f"验证失败: {result.error}")
    """

    def __init__(self, project_dir: Optional[Path] = None):
        """初始化评估编排器。

        Args:
            project_dir: ultralytics project 目录（默认 RUNS_DIR / "val"）
        """
        self.project_dir = Path(project_dir) if project_dir else (RUNS_DIR / "val")

    def _generate_experiment_name(self, prefix: str, model_stem: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        existing = list(self.project_dir.glob(f"{prefix}-*"))
        counter = len(existing) + 1
        return f"{prefix}-{counter}_{timestamp}_{model_stem}"

    def validate(
        self,
        model_path: str,
        dataset: str,
        yaml_path: Optional[str] = None,
        task: str = "detect",
        split: str = "val",
        cli_overrides: Optional[dict] = None,
        name: Optional[str] = None,
        dry_run: bool = False,
        save_json: bool = False,
    ) -> ValResult:
        """执行模型评估。

        Args:
            model_path: 模型 .pt 文件路径
            dataset: 数据集名称（用于查找 configs/datasets/<name>.yaml）
            yaml_path: 验证配置 YAML 路径（不指定则自动生成）
            task: 算法任务类型 (detect/segment/classify)
            split: 验证数据集划分 (train/val/test)
            cli_overrides: CLI 覆盖参数（conf/iou/device/batch/imgsz/max_det/half）
            name: 实验名称（不指定则自动生成）
            dry_run: 仅生成配置，不实际验证
            save_json: 保存详细验证结果为 JSON

        Returns:
            ValResult: 验证结果
        """
        model_path_obj = Path(model_path).resolve()
        model_stem = model_path_obj.stem.replace("-best", "").replace("-last", "")
        experiment_name = name or self._generate_experiment_name("val", model_stem)
        val_project_dir = self.project_dir
        val_project_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"ODPlatform 模型评估 — {experiment_name}")
        logger.info(f"  模型: {model_path_obj.name}")
        logger.info(f"  数据集: {dataset}")
        logger.info(f"  划分: {split}")
        logger.info("=" * 60)

        # ── Step 1: 运行配置 ──
        logger.info("[1/3] 运行配置")

        all_overrides = dict(cli_overrides or {})
        all_overrides["model"] = str(model_path_obj)

        snapshot_run_dir = RUNS_DIR / "run_config" / experiment_name
        snapshot_run_dir.mkdir(parents=True, exist_ok=True)

        if yaml_path:
            config_file = Path(yaml_path)
            if not config_file.is_absolute():
                config_file = APP_DIR / config_file
            logger.info(f"    加载配置: {config_file}")
            bundle = build_config(
                task=task,
                yaml_path=config_file,
                cli_args=all_overrides if all_overrides else None,
            )
        else:
            logger.info("    自动生成验证配置")
            auto_config = snapshot_run_dir / "val_config.yaml"
            generate_template_to_file(task="val", output_path=auto_config, force=True)
            bundle = build_config(
                task=task,
                yaml_path=auto_config,
                cli_args=all_overrides if all_overrides else None,
            )

        if not bundle.valid:
            logger.error("    配置验证失败:")
            for err in bundle.errors:
                logger.error(f"      [ERROR] {err.field}: {err.message}")
            return ValResult(success=False, experiment_name=experiment_name,
                             error=f"配置验证失败: {bundle.errors}")

        snapshot = ConfigSnapshot.from_bundle(bundle)
        snapshot_path = snapshot_run_dir / "config_snapshot.json"
        save_snapshot_to_file(snapshot, snapshot_path)

        ultralytics_args = bundle.to_ultralytics_args()
        logger.info(f"    Ultralytics 参数: {json.dumps(ultralytics_args, ensure_ascii=False)}")

        # ── Step 2: 数据集检查 ──
        data_yaml = CONFIGS_DATASETS_DIR / f"{dataset}.yaml"
        if not data_yaml.exists():
            logger.error(f"    数据集 YAML 不存在: {data_yaml}")
            avail = [p.stem for p in CONFIGS_DATASETS_DIR.glob("*.yaml")]
            logger.info(f"    可用数据集: {avail}")
            return ValResult(success=False, experiment_name=experiment_name,
                             error=f"数据集 {dataset} 不存在, 可用: {avail}",
                             snapshot_path=snapshot_path)

        # ── Step 3: 执行验证 ──
        if dry_run:
            logger.info("[2/3] 跳过验证（--dry-run）")
            logger.info(f"\n实验 '{experiment_name}' 完整配置:")
            logger.info(f"  配置快照: {snapshot_path}")
            logger.info(f"\n直接验证命令:")
            cmd_parts = [
                "yolo",
                f"task={task}",
                f"mode=val",
                f"model={model_path_obj}",
                f"data={data_yaml}",
            ]
            for k, v in ultralytics_args.items():
                if k in ("model",):
                    continue
                cmd_parts.append(f"{k}={v}")
            logger.info("  " + " ".join(cmd_parts))
            return ValResult(success=True, experiment_name=experiment_name,
                             output_dir=val_project_dir / experiment_name,
                             snapshot_path=snapshot_path)

        logger.info(f"[2/3] 启动验证 — 模型: {model_path_obj.name}")
        logger.info(f"    数据: {data_yaml.name}")
        logger.info(f"    输出: {val_project_dir / experiment_name}")

        try:
            model = YOLO(str(model_path_obj))
            results = model.val(
                data=str(data_yaml),
                project=str(val_project_dir),
                name=experiment_name,
                split=split,
                save_json=save_json,
                **ultralytics_args,
            )

            rd = results.results_dict if hasattr(results, 'results_dict') else {}
            metrics = {
                "map50":    float(rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", 0))),
                "map50_95": float(rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", 0))),
                "precision": float(rd.get("metrics/precision(B)", rd.get("metrics/precision", 0))),
                "recall":   float(rd.get("metrics/recall(B)", rd.get("metrics/recall", 0))),
                "fitness":  float(rd.get("fitness", 0)),
            }
            val_dir = val_project_dir / experiment_name

            metrics_path = val_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            val_summary = {
                "experiment": experiment_name,
                "model": str(model_path_obj),
                "dataset": dataset,
                "split": split,
                "task": task,
                "metrics": metrics,
                "output_dir": str(val_dir),
                "snapshot": str(snapshot_path),
                "timestamp": datetime.now().isoformat(),
                "success": True,
            }
            summary_path = snapshot_run_dir / "val_summary.json"
            summary_path.write_text(
                json.dumps(val_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            logger.info(f"    ✓ 评估完成")
            logger.info(f"    输出目录: {val_dir}")
            logger.info(f"    指标 JSON: {metrics_path}")
            logger.info(f"    评估摘要: {summary_path}")
            logger.info(f"    mAP50:    {metrics['map50']:.4f}")
            logger.info(f"    mAP50-95: {metrics['map50_95']:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")

            return ValResult(
                success=True,
                experiment_name=experiment_name,
                output_dir=val_dir,
                metrics=metrics,
                snapshot_path=snapshot_path,
            )

        except KeyboardInterrupt:
            logger.warning("评估被用户中断")
            return ValResult(success=False, experiment_name=experiment_name,
                             error="用户中断", snapshot_path=snapshot_path)
        except Exception as e:
            logger.exception(f"评估失败: {e}")
            error_summary = {
                "experiment": experiment_name,
                "model": str(model_path_obj),
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
            }
            summary_path = snapshot_run_dir / "val_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(error_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return ValResult(success=False, experiment_name=experiment_name,
                             error=str(e), snapshot_path=snapshot_path)

    def _ensure_dir(self, d: Path) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        return d
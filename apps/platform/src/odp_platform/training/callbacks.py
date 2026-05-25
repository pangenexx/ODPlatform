from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR, ROOT_DIR

logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="train",
    logger_name="odp-train",
)

_WEB_BACKEND_DIR: Path = ROOT_DIR / "apps" / "web-backend"
if str(_WEB_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_BACKEND_DIR))

_BACKEND_AVAILABLE: bool = False
_hooks = None
try:
    from hooks import (
        on_epoch_end as _on_epoch_end,
        on_training_end as _on_training_end,
        on_training_failed as _on_training_failed,
        on_training_start as _on_training_start,
    )
    _BACKEND_AVAILABLE = True
    logger.info("后端 hooks 已加载，训练进度将同步到数据库")
except ImportError:
    logger.info("后端 hooks 不可用（跳过），训练不受影响")


class TrainingHooks:
    def __init__(self, backend_url: str = "http://127.0.0.1:8000") -> None:
        self._exp_id: Optional[int] = None
        self._backend_url: str = backend_url
        self._is_available: bool = _BACKEND_AVAILABLE

    @property
    def exp_id(self) -> Optional[int]:
        return self._exp_id

    @property
    def is_available(self) -> bool:
        return self._is_available

    def on_train_start(
        self,
        name: str,
        config_json: str,
        dataset: str,
        model: str,
    ) -> Optional[int]:
        if not self._is_available:
            return None
        exp_id = _on_training_start(
            name=name,
            config_json=config_json,
            dataset=dataset,
            model=model,
            base_url=self._backend_url,
        )
        self._exp_id = exp_id
        return exp_id

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        map50: float = 0.0,
        map50_95: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        lr: float = 0.0,
    ) -> None:
        if not self._is_available or self._exp_id is None:
            return
        _on_epoch_end(
            experiment_id=self._exp_id,
            epoch=epoch,
            metrics={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "map50": map50,
                "map50_95": map50_95,
                "precision": precision,
                "recall": recall,
                "lr": lr,
            },
            base_url=self._backend_url,
        )

    def on_epoch_end_from_csv_row(self, row: dict[str, str]) -> None:
        epoch = int(row.get("epoch", 0))
        if epoch == 0:
            return
        train_loss = (
            float(row.get("train/box_loss", 0))
            + float(row.get("train/cls_loss", 0))
            + float(row.get("train/dfl_loss", 0))
        )
        val_loss = (
            float(row.get("val/box_loss", 0))
            + float(row.get("val/cls_loss", 0))
            + float(row.get("val/dfl_loss", 0))
        )
        self.on_epoch_end(
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            map50=round(float(row.get("metrics/mAP50(B)", 0)), 4),
            map50_95=round(float(row.get("metrics/mAP50-95(B)", 0)), 4),
            precision=round(float(row.get("metrics/precision(B)", 0)), 4),
            recall=round(float(row.get("metrics/recall(B)", 0)), 4),
            lr=float(row.get("lr/pg0", 0)),
        )

    def on_train_end(self, best_map50: float, model_path: str) -> None:
        if not self._is_available or self._exp_id is None:
            return
        _on_training_end(
            experiment_id=self._exp_id,
            map50=best_map50,
            model_path=model_path,
            base_url=self._backend_url,
        )

    def on_train_failed(self, reason: str = "") -> None:
        if not self._is_available or self._exp_id is None:
            return
        _on_training_failed(
            experiment_id=self._exp_id,
            reason=reason,
            base_url=self._backend_url,
        )
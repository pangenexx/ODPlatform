from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

import requests

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import (
    CHECKPOINTS_DIR,
    LOGGING_DIR,
)

logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="train",
    logger_name="odp-train",
)


# =========================================================
# Backend Config
# =========================================================

BACKEND_URL = "http://127.0.0.1:8000"

MAX_RETRIES = 3
RETRY_DELAY = 1.0


# =========================================================
# HTTP Utils
# =========================================================

def _post_with_retry(
    url: str,
    json_data: dict,
    label: str = "",
) -> Optional[dict]:
    """
    带重试 POST

    后端未启动时：
        记录 warning
        不中断训练
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            r = requests.post(
                url,
                json=json_data,
                timeout=5,
            )

            r.raise_for_status()

            return r.json()

        except requests.exceptions.RequestException as e:

            logger.warning(
                f"[{label}] "
                f"后端通信失败 "
                f"(尝试 {attempt}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logger.error(
        f"[{label}] "
        f"重试耗尽，放弃同步"
    )

    return None


# ---------------------------------------------------------

def _patch_with_retry(
    url: str,
    json_data: dict,
    label: str = "",
) -> Optional[dict]:
    """
    带重试 PATCH
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            r = requests.patch(
                url,
                json=json_data,
                timeout=5,
            )

            r.raise_for_status()

            return r.json()

        except requests.exceptions.RequestException as e:

            logger.warning(
                f"[{label}] "
                f"后端 PATCH 失败 "
                f"(尝试 {attempt}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logger.error(
        f"[{label}] "
        f"PATCH 重试耗尽"
    )

    return None


# =========================================================
# EarlyStopping
# =========================================================

class EarlyStopping:
    """
    EarlyStopping

    根据验证集 map50 自动停止训练
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.best_score = -1.0

        self.counter = 0

        self.should_stop = False

    # -----------------------------------------------------

    def step(
        self,
        score: float,
    ) -> bool:
        """
        更新状态

        Returns:
            bool:
                是否应该停止训练
        """

        # 第一次
        if self.best_score < 0:

            self.best_score = score

            logger.info(
                f"初始化 best score: {score:.6f}"
            )

            return False

        # 有提升
        if score > self.best_score + self.min_delta:

            logger.info(
                f"metric improved: "
                f"{self.best_score:.6f} "
                f"-> {score:.6f}"
            )

            self.best_score = score

            self.counter = 0

            return False

        # 无提升
        self.counter += 1

        logger.info(
            f"metric not improved "
            f"({self.counter}/{self.patience})"
        )

        # 触发停止
        if self.counter >= self.patience:

            self.should_stop = True

            logger.warning(
                "触发 EarlyStopping"
            )

            return True

        return False


# =========================================================
# Best Checkpoint
# =========================================================

class BestCheckpointManager:
    """
    自动保存 best mAP checkpoint
    """

    def __init__(
        self,
        experiment_name: str,
    ):
        self.experiment_name = experiment_name

        self.best_map50 = -1.0

        self.best_model_path: Optional[Path] = None

    # -----------------------------------------------------

    def update(
        self,
        map50: float,
        current_checkpoint: str | Path,
    ) -> bool:
        """
        更新 best checkpoint

        Returns:
            bool:
                是否更新了 best
        """

        current_checkpoint = Path(
            current_checkpoint
        )

        # 当前 checkpoint 不存在
        if not current_checkpoint.exists():

            logger.warning(
                f"checkpoint 不存在: "
                f"{current_checkpoint}"
            )

            return False

        # 没提升
        if map50 <= self.best_map50:

            return False

        # 更新 best
        self.best_map50 = map50

        target_path = (
            CHECKPOINTS_DIR
            / f"best_{self.experiment_name}.pt"
        )

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy(
            current_checkpoint,
            target_path,
        )

        self.best_model_path = target_path

        logger.info(
            f"best checkpoint 更新: "
            f"mAP50={map50:.6f} "
            f"-> {target_path}"
        )

        return True


# =========================================================
# Training Hooks
# =========================================================

class TrainingHooks:
    """
    训练回调系统

    功能：
        - backend 通信
        - epoch metrics 同步
        - 训练状态同步
    """

    def __init__(
        self,
        experiment_name: str,
        config_json: str,
    ):
        self.exp_name = experiment_name

        self.config_json = config_json

        self.exp_id: Optional[int] = None

    # =====================================================
    # train start
    # =====================================================

    def on_train_start(
        self,
        dataset: str,
        model: str,
    ):
        """
        训练开始

        POST:
            /api/experiments
        """

        payload = {
            "name":
                self.exp_name,

            "config_json":
                self.config_json,

            "dataset":
                dataset,

            "model":
                model,
        }

        result = _post_with_retry(
            f"{BACKEND_URL}/api/experiments",
            payload,
            label="train_start",
        )

        if result:

            self.exp_id = result.get("id")

            logger.info(
                f"实验注册成功: "
                f"id={self.exp_id}"
            )

    # =====================================================
    # epoch end
    # =====================================================

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict,
    ):
        """
        epoch 结束

        POST:
            /api/experiments/{id}/epochs
        """

        if self.exp_id is None:

            logger.warning(
                "exp_id is None，"
                "跳过 epoch 同步"
            )

            return

        payload = {
            "epoch":
                epoch,

            **metrics,
        }

        _post_with_retry(
            f"{BACKEND_URL}/api/experiments/"
            f"{self.exp_id}/epochs",

            payload,

            label=f"epoch_{epoch}",
        )

    # =====================================================
    # train end
    # =====================================================

    def on_train_end(
        self,
        map50: float,
        model_path: str,
    ):
        """
        训练结束

        PATCH:
            /api/experiments/{id}/status
        """

        if self.exp_id is None:

            logger.warning(
                "exp_id is None，"
                "跳过 train_end 同步"
            )

            return

        payload = {
            "status":
                "completed",

            "best_map50":
                map50,

            "model_path":
                model_path,
        }

        result = _patch_with_retry(
            f"{BACKEND_URL}/api/experiments/"
            f"{self.exp_id}/status",

            payload,

            label="train_end",
        )

        if result:

            logger.info(
                f"实验完成同步: "
                f"id={self.exp_id}"
            )


# =========================================================
# Combined Callback Manager
# =========================================================

class CallbackManager:
    """
    统一 callback manager

    聚合：
        - EarlyStopping
        - BestCheckpoint
        - Backend Hooks
    """

    def __init__(
        self,
        experiment_name: str,
        config_json: str,
        patience: int = 50,
    ):
        self.hooks = TrainingHooks(
            experiment_name,
            config_json,
        )

        self.early_stopping = EarlyStopping(
            patience=patience,
        )

        self.checkpoint_manager = (
            BestCheckpointManager(
                experiment_name
            )
        )

    # -----------------------------------------------------

    def on_train_start(
        self,
        dataset: str,
        model: str,
    ):
        self.hooks.on_train_start(
            dataset,
            model,
        )

    # -----------------------------------------------------

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict,
        checkpoint_path: str | Path,
    ) -> bool:
        """
        epoch 结束统一处理

        Returns:
            bool:
                是否应该停止训练
        """

        # backend sync
        self.hooks.on_epoch_end(
            epoch,
            metrics,
        )

        # checkpoint
        self.checkpoint_manager.update(
            metrics.get("map50", 0),
            checkpoint_path,
        )

        # early stopping
        should_stop = (
            self.early_stopping.step(
                metrics.get("map50", 0)
            )
        )

        return should_stop

    # -----------------------------------------------------

    def on_train_end(
        self,
        map50: float,
        model_path: str,
    ):
        self.hooks.on_train_end(
            map50,
            model_path,
        )


# =========================================================
# Debug
# =========================================================

if __name__ == "__main__":

    callbacks = CallbackManager(
        experiment_name="debug_exp",
        config_json='{"epochs": 100}',
        patience=3,
    )

    callbacks.on_train_start(
        dataset="rsod",
        model="yolo11n.pt",
    )

    scores = [
        0.50,
        0.55,
        0.57,
        0.56,
        0.55,
        0.54,
    ]

    for epoch, score in enumerate(scores):

        metrics = {
            "map50": score,
            "precision": 0.8,
            "recall": 0.7,
        }

        stop = callbacks.on_epoch_end(
            epoch=epoch,
            metrics=metrics,
            checkpoint_path="dummy.pt",
        )

        print(
            f"epoch={epoch} "
            f"map50={score:.4f} "
            f"stop={stop}"
        )

        if stop:
            break

    callbacks.on_train_end(
        map50=0.57,
        model_path="best_debug.pt",
    )
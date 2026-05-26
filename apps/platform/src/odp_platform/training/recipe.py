from __future__ import annotations

from copy import deepcopy

from odp_platform.training.experiment import (
    ExperimentConfig,
)

# =========================================================
# RSOD Baseline
# =========================================================

RSOD_BASELINE = ExperimentConfig(
    name="rsod_baseline",

    dataset="rsod",

    model="yolo11n.pt",

    task="detect",

    epochs=100,

    batch=16,

    imgsz=640,

    lr0=0.01,

    device="",

    workers=2,

    optimizer="auto",

    amp=True,

    patience=50,

    seed=42,

    note="RSOD baseline experiment",
)

# =========================================================
# VisDrone Baseline
# =========================================================

VISDRONE_BASELINE = ExperimentConfig(
    name="visdrone_baseline",

    dataset="visdrone",

    model="yolo11n.pt",

    task="detect",

    epochs=150,

    batch=8,

    imgsz=1024,

    lr0=0.005,

    device="",

    workers=2,

    optimizer="auto",

    amp=True,

    patience=50,

    seed=42,

    note="VisDrone baseline experiment",
)

# =========================================================
# LR Sweep
# =========================================================

LR_SWEEP = [
    ExperimentConfig(
        name=f"rsod_lr_{lr}",

        dataset="rsod",

        model="yolo11n.pt",

        epochs=100,

        batch=16,

        imgsz=640,

        lr0=lr,

        note=f"LR sweep lr={lr}",
    )

    for lr in [
        0.1,
        0.01,
        0.001,
        0.0001,
    ]
]

# =========================================================
# Image Size Sweep
# =========================================================

IMGSZ_SWEEP = [
    ExperimentConfig(
        name=f"rsod_imgsz_{imgsz}",

        dataset="rsod",

        model="yolo11n.pt",

        epochs=100,

        batch=16,

        imgsz=imgsz,

        lr0=0.01,

        note=f"Image size sweep imgsz={imgsz}",
    )

    for imgsz in [
        640,
        800,
        1024,
    ]
]

# =========================================================
# Batch Size Sweep
# =========================================================

BATCH_SWEEP = [
    ExperimentConfig(
        name=f"rsod_batch_{batch}",

        dataset="rsod",

        model="yolo11n.pt",

        epochs=100,

        batch=batch,

        imgsz=640,

        lr0=0.01,

        note=f"Batch size sweep batch={batch}",
    )

    for batch in [
        4,
        8,
        16,
    ]
]

# =========================================================
# Small Object Experiments
# =========================================================

SMALL_OBJECT_RECIPES = [
    ExperimentConfig(
        name="rsod_yolo11n_p2",

        dataset="rsod",

        model="training/experiments/"
              "small_object/yolo11n_p2.yaml",

        epochs=120,

        batch=16,

        imgsz=1024,

        lr0=0.005,

        note="P2 head for small object detection",
    ),

    ExperimentConfig(
        name="rsod_yolo11n_cbam",

        dataset="rsod",

        model="training/experiments/"
              "attention/yolo11n_cbam.yaml",

        epochs=120,

        batch=16,

        imgsz=1024,

        lr0=0.005,

        note="CBAM attention experiment",
    ),
]

# =========================================================
# Factory Functions
# =========================================================

def clone_recipe(
    base: ExperimentConfig,
    new_name: str,
    **kwargs,
) -> ExperimentConfig:
    """
    基于已有 recipe 复制新实验
    """

    cfg = deepcopy(base)

    cfg.name = new_name

    for k, v in kwargs.items():

        setattr(cfg, k, v)

    return cfg


# ---------------------------------------------------------

def make_lr_sweep(
    base_config: ExperimentConfig,
    lr_values: list[float],
) -> list[ExperimentConfig]:
    """
    动态生成 LR sweep
    """

    recipes = []

    for lr in lr_values:

        cfg = clone_recipe(
            base_config,

            new_name=f"{base_config.name}_lr_{lr}",

            lr0=lr,

            note=f"Auto generated lr sweep: {lr}",
        )

        recipes.append(cfg)

    return recipes


# ---------------------------------------------------------

def make_imgsz_sweep(
    base_config: ExperimentConfig,
    imgsz_values: list[int],
) -> list[ExperimentConfig]:
    """
    动态生成 imgsz sweep
    """

    recipes = []

    for imgsz in imgsz_values:

        cfg = clone_recipe(
            base_config,

            new_name=f"{base_config.name}_imgsz_{imgsz}",

            imgsz=imgsz,

            note=f"Auto generated imgsz sweep: {imgsz}",
        )

        recipes.append(cfg)

    return recipes


# ---------------------------------------------------------

def make_ablation_recipes(
    base_config: ExperimentConfig,
    model_paths: list[str],
) -> list[ExperimentConfig]:
    """
    消融实验生成器
    """

    recipes = []

    for model_path in model_paths:

        name = model_path.split("/")[-1]
        name = name.replace(".yaml", "")

        cfg = clone_recipe(
            base_config,

            new_name=f"{base_config.name}_{name}",

            model=model_path,

            note=f"Ablation experiment: {name}",
        )

        recipes.append(cfg)

    return recipes


# =========================================================
# All Recipes Registry
# =========================================================

ALL_RECIPES = {
    "rsod_baseline":
        RSOD_BASELINE,

    "visdrone_baseline":
        VISDRONE_BASELINE,

    "lr_sweep":
        LR_SWEEP,

    "imgsz_sweep":
        IMGSZ_SWEEP,

    "batch_sweep":
        BATCH_SWEEP,

    "small_object":
        SMALL_OBJECT_RECIPES,
}

# =========================================================
# Debug
# =========================================================

if __name__ == "__main__":

    print("=" * 80)
    print("RSOD BASELINE")
    print("=" * 80)

    print(RSOD_BASELINE)

    print()

    print("=" * 80)
    print("LR SWEEP")
    print("=" * 80)

    for cfg in LR_SWEEP:

        print(
            cfg.name,
            cfg.lr0,
        )
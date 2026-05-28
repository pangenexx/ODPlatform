# ODPlatform - 全流程学习指南

> 作者: MuU | 目标: 从零掌握目标检测全栈开发技术路径
> 版本: v2.0 (全面审计修正版)
> **命令速查请见**：[ODPlatform_命令速查.md](ODPlatform_命令速查.md)

---

## 1. 项目全景概览

### 1.1 这是什么项目？

ODPlatform 是一个**基于 YOLO 的多格式目标检测开发平台**，覆盖从数据处理到模型推理的完整流程。

```
数据输入 → 格式转换 → 数据校验 → 模型训练 → 模型评估 → 模型推理
  VOC/COCO   YOLO格式    验证清洗    YOLO训练   mAP计算    单图/批量
  LabelMe    统一管理      可视化     调参       结果分析    流水线
```

核心特点：
- **双入口**：CLI 命令 + Gradio WebUI（用户/管理员双模式）
- **双后端**：FastAPI + SQLite 后端（可选）用于持久化训练/推理记录
- **模块化架构**：CLI 直调各业务模块，无中间抽象层
- **渐进演进**：Stage 0 (marker) → Stage 1 (apps/platform 单体) → Stage 2 (前后端分离)

### 1.2 完整目录结构

```
ODPlatform/
├── apps/
│   ├── platform/                    ← 核心引擎（全量代码）
│   │   ├── configs/datasets/          数据集 YAML 配置
│   │   ├── logging/                   运行时日志
│   │   ├── src/odp_platform/
│   │   │   ├── _version.py            版本号（单一数据源）
│   │   │   ├── common/                基础设施（路径/日志/性能/常量）
│   │   │   ├── data_pipeline/         数据格式转换 + 数据集划分
│   │   │   ├── data_validation/       数据质量校验
│   │   │   ├── training/              模型训练（experiment + hooks + recipe）
│   │   │   ├── evaluation/            模型评估
│   │   │   ├── inference/             模型推理（engine + visualizer + frame_source）
│   │   │   ├── run_config/            配置管理（loader/merger/validator/schema）
│   │   │   ├── webui/                 Gradio 前端
│   │   │   └── cli/                   10 个 CLI 入口命令
│   │   └── pyproject.toml             构建配置 + [project.scripts]
│   │
│   └── web-backend/                  ← FastAPI 后端（可选）
│       ├── main.py                     FastAPI app
│       ├── api/                        API 路由（experiments/detection/models/auth...）
│       ├── db/                         数据库（SQLite + SQLAlchemy）
│       ├── schemas.py                  Pydantic 数据模型
│       └── hooks.py                    废弃（转用 training/callbacks.py）
│
├── data/                            ← 数据集/模型/运行产物（gitignore）
│   ├── raw/                            原始标注数据
│   ├── models/checkpoints/             训练好的 .pt 权重
│   ├── runs/experiments/               训练实验产物
│   └── ...
├── docs/architecture/               ← ADR 架构决策记录
├── scripts/                          ← 运维脚本
└── pyproject.toml                    ← 顶层 workspace（dev 工具配置）
```

---

## 2. 模块详解：各司其职

### 2.1 common/ — 基础设施层

**路径**：`apps/platform/src/odp_platform/common/`

```python
from odp_platform.common.paths import ROOT_DIR, DATA_DIR, CHECKPOINTS_DIR, ...
```

| 文件 | 功能 |
|------|------|
| `paths.py` | `.odp-workspace` marker 向上遍历找根目录，统一管理所有路径常量 |
| `logging_utils.py` | 根 Logger 装配 + 彩色控制台 + 文件日志，幂等保护 |
| `constants.py` | 共享枚举（`AnnotationFormat`, `Task`, `RunTask`）和字面量 |
| `performance_utils.py` | `@time_it` 装饰器、`timer` 上下文管理器、`MetricTracker` 等 |
| `string_utils.py` | 格式化表格工具（`format_table_row`） |
| `system_utils.py` | `log_device_info()` 打印系统/GPU 信息 |

**设计要点**：
- `paths.py` 不硬编码任何绝对路径，全部从 marker 动态探测
- `logging_utils.py` 有幂等保护：`if logger.handlers: return logger`
- `constants.py` 用 class 定义枚举（非 Enum），方便装饰器直接引用

### 2.2 data_pipeline/ — 数据格式转换

**路径**：`apps/platform/src/odp_platform/data_pipeline/`

**核心设计——注册表模式**：

```
registry.py            ← `_REGISTRY` dict + `@register` 装饰器
  │
  ├── service.py       ← `converter_data_to_yolo()` 查询注册表并调用
  │
  ├── core/            ← 具体转换器（被 @register 自动注册）
  │   ├── coco.py          COCO JSON → YOLO txt（调 ultralytics 实现）
  │   ├── pascal_voc.py    Pascal VOC XML → YOLO txt
  │   └── yolo.py          YOLO txt 之间的格式转换/校验
  │
  ├── split/           ← 数据集划分
  │   ├── splitter.py      图片-标签对按比例随机划分
  │   ├── materializer.py  划分结果落盘（创建 train/val/test 目录）
  │   ├── yaml_writer.py   生成 <dataset>.yaml（供 YOLO 训练使用）
  │   └── manifest.py      PairList：图片-标签配对数据结构
  │
  └── orchestrator.py  ← DatasetPipeline 端到端编排器
```

**端到端流程**：

```
CLI: odp-transform --dataset rsod --format pascal_voc
  │
  ├── DatasetPipeline.__init__(dataset_name="rsod", annotation_format="pascal_voc")
  │
  ├── _check_raw() → 验证 data/raw/rsod/{images,annotations}/ 存在
  │
  ├── service.converter_data_to_yolo() → registry.get_converter("pascal_voc")
  │   └── core/pascal_voc.py 被 @register("pascal_voc") 装饰 → 自动注册
  │   └── entry.func(input_dir, output_labels_dir, options)
  │       └── 解析 XML → 提取 bbox + class → 写入 YOLO txt
  │
  ├── split_pairs() → 按 train_rate/val_rate 随机打乱划分
  │
  ├── materialize() → 创建 data/train/{images,labels}/ + val/ + test/
  │   └── 软链接图片 + 写入 .txt 标签
  │
  └── write_dataset_yaml() → configs/datasets/rsod.yaml
      └── 包含: path/nc/names/train/val/test
```

**支持格式**：

| 输入格式 | 注册名 | 支持任务 |
|---------|--------|---------|
| Pascal VOC (XML) | `pascal_voc` | detect |
| COCO (JSON) | `coco` | detect, segment |
| YOLO (txt) | `yolo` | detect |

新增格式只需 `@register("new_format")` 实现一个转换函数，无需改任何其他代码。

### 2.3 data_validation/ — 数据质量校验

**路径**：`apps/platform/src/odp_platform/data_validation/`

同样采用注册表模式：

```
registry.py ← `_REGISTRY` + `@check("name")` 装饰器
  │
  ├── service.py     ← validate_dataset() 端到端入口
  ├── report.py      ← ValidationReport 数据类、render_to_logger()
  ├── snapshot.py    ← DatasetSnapshot 数据快照（YAML + 目录扫描）
  ├── render.py      ← 渲染工具
  │
  └── checks/        ← 具体检查项（被 @check 自动注册）
      ├── yaml_schema.py     ← 检查 nc/path/names/train/val/test 字段
      ├── pair_existence.py  ← 检查每张图是否有对应标签文件
      ├── label_format.py    ← 检查标签内容格式（YOLO: cls x y w h）
      └── split_uniqueness.py ← 检查 train/val/test 无重复
```

**端到端流程**：

```
CLI: odp-validate --dataset rsod
  │
  ├── validate_dataset(yaml_path)
  │   ├── build_snapshot() → 解析 YAML + 扫描目录
  │   │   └── 收集 images_per_split / labels_per_split 信息
  │   │
  │   ├── run_all_checks()
  │   │   ├── yaml_schema → 验证 nc/path/names 字段
  │   │   ├── pair_existence → 验证图片-标签对应关系
  │   │   ├── label_format → 验证每行 cls x y w h 格式
  │   │   └── split_uniqueness → 验证跨 split 无重复
  │   │
  │   └── ValidationReport → JSON + 控制台输出
  │
  └── 返回 exit_code（0=通过, 1=警告, 2=错误）
```

### 2.4 training/ — 模型训练

**路径**：`apps/platform/src/odp_platform/training/`

| 文件 | 功能 |
|------|------|
| `experiment.py` | `ExperimentConfig` + `run_experiment()` + `ExperimentResult` |
| `callbacks.py` | `TrainingHooks`（后端同步）+ `normalize_csv_row()` |
| `tracker.py` | `ExperimentSummary` 数据结构 |
| `recipe.py` | 预置实验配置（RSOD/VisDrone baseline, LR sweep） |

**端到端训练流程**：

```
CLI: odp-train --dataset rsod --model yolo11n.pt --epochs 100
  │
  ├── cli/train.py 解析参数 → build_config() 生成配置 → 合并 CLI 覆盖
  │
  ├── run_experiment(config)
  │   ├── 创建实验目录 runs/experiments/<name>/
  │   ├── 保存 config_snapshot.json 快照
  │   │
  │   ├── TrainingHooks.on_train_start()
  │   │   └── POST /api/experiments （后端不可达则静默降级）
  │   │
  │   ├── YOLO.train(**train_kwargs)
  │   │   ├── data = configs/datasets/<dataset>.yaml
  │   │   ├── epochs/batch/imgsz/lr0/optimizer/amp/patience
  │   │   └── project = runs/experiments/, name = <name>
  │   │   │
  │   │   └── 训练过程中，Ultralytics 自动回调：
  │   │       └── TrainingHooks.on_epoch_end() → normalize_csv_row() → POST epochs
  │   │
  │   ├── TrainingHooks.on_train_end()
  │   │   └── PATCH /api/experiments/{id}/status
  │   │
  │   ├── 复制 best.pt → checkpoints/best_<name>.pt
  │   │
  │   ├── _sync_to_backend() → POST /api/experiments
  │   │
  │   └── 返回 ExperimentResult（map50/map50_95/precision/recall/...）
  │
  └── 输出：
      ├── runs/experiments/<name>/
      │   ├── config_snapshot.json
      │   ├── results.csv（逐轮指标）
      │   ├── results.png / PR_curve.png / F1_curve.png
      │   ├── confusion_matrix.png / labels.jpg
      │   └── weights/best.pt + last.pt
      └── data/models/checkpoints/best_<name>.pt
```

**预置实验配方**（`recipe.py`）：

| 配方名 | 数据集 | 模型 | Epochs | ImgSz | Batch | LR |
|--------|--------|------|--------|-------|-------|----|
| RSOD_BASELINE | rsod | yolo11n.pt | 100 | 640 | 16 | 0.01 |
| VISDRONE_BASELINE | visdrone | yolo11n.pt | 150 | 1024 | 8 | 0.005 |
| LR_SWEEP | rsod | yolo11n.pt | 100 | 640 | 16 | 0.01/0.005/0.001 |

**CSV 列名适配器**（`callbacks.py`）：

```python
_COLUMN_ALIASES = {
    "metrics/mAP50(B)": "map50",      # Ultralytics v8 格式
    "metrics/mAP50-95(B)": "map50_95",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "train/box_loss": "box_loss",
    "val/box_loss": "val_box_loss",
    "lr/pg0": "lr",
}
```

### 2.5 run_config/ — 配置管理

**路径**：`apps/platform/src/odp_platform/run_config/`

独立于业务链的配置子系统，处理四层配置覆盖：

```
CLI 参数 (最高优先级)
     ↓
  YAML 配置文件
     ↓
  环境变量 ODP_*
     ↓
  默认值 (最低优先级)
```

**目录结构**：

```
run_config/
├── __init__.py        导出 API：build_config, save_snapshot, restore_from_snapshot
├── registry.py        @config_generator 装饰器 + list_fields()
├── service.py         核心调度 (build_config / validate / trace / snapshot)
├── loader.py          load_yaml / parse_cli_args / resolve_yaml_path
├── merger.py          多源配置合并 + TraceReport
├── schema.py          ConfigBundle / ConfigSnapshot / TraceRecord 数据类
├── validator.py       配置校验（字段类型/范围/必填）
├── template.py        配置模板生成
└── fields/            各任务字段定义
    ├── train.py       训练配置字段
    ├── val.py         评估配置字段
    └── predict.py     推理配置字段
```

**关键 API**：

```python
from odp_platform.run_config import build_config, save_snapshot_to_file

# 构建配置（YAML + CLI 覆盖 + 环境变量）
bundle = build_config(task="train", yaml_path=Path("train.yaml"), cli_args={"epochs": 200})
bundle.config    # 合并后的完整配置 dict
bundle.trace     # TraceReport：每个字段的来源追溯
bundle.errors    # 验证错误列表
bundle.warnings  # 验证警告列表

# 快照
save_snapshot_to_file(bundle.snapshot, Path("snapshot.json"))
restored = restore_from_snapshot(loaded_snapshot)
```

### 2.6 evaluation/ — 模型评估

**路径**：`apps/platform/src/odp_platform/evaluation/`

| 文件 | 功能 |
|------|------|
| `service.py` | `ValService.validate()` 使用 Ultralytics YOLO 验证模型精度 |

**端到端流程**：

```
CLI: odp-val --model best.pt --dataset rsod
  │
  ├── ValService.validate(model_path, dataset, yaml_path)
  │   ├── build_config(task="val") → 配置合并
  │   ├── YOLO.val(data=dataset_yaml, model=model_path, ...)
  │   └── 返回 ValResult(metrics = {map50, map50_95, precision, recall, ...})
  │
  └── 输出 runs/val/<experiment_name>/
      ├── config_snapshot.json
      └── 验证结果指标
```

### 2.7 inference/ — 模型推理

**路径**：`apps/platform/src/odp_platform/inference/`

推理模块是整个项目中最复杂的模块，分为三层：

```
业务层:
  service.py           ← InferService.predict() 端到端推理编排
  engine.py            ← Detector：YOLO 模型加载 + 检测
  pipeline_config.py   ← PipelineConfig：推理管道配置
  visualizer.py        ← draw_detections / draw_info_panel / draw_confidence_histogram
  benchmark.py         ← 推理基准测试
  data_visualizer.py   ← 数据可视化工具
  utils.py             ← 推理工具函数

帧源抽象层 (frame_source/):
  core/
    base.py            ← FrameSource 抽象基类（open/read/close/seek）
    config.py          ← CameraConfig (Pydantic)
    types.py           ← Frame/SourceType 类型定义
  sources/
    camera.py          ← CameraSource（OpenCV 摄像头）
    image.py           ← ImageSource / ImageFolderSource
    video.py           ← VideoSource
  wrappers/
    threaded.py        ← ThreadedSource（采集-消费分离，实时推理首选）
    aio.py             ← AsyncSource（异步接口）
  factory.py           ← create_frame_source() + 注册表模式
  overlay.py           ← HUD 叠加层（FPS/帧率/检测数）
```

**端到端推理流程**：

```
CLI: odp-infer --source test.jpg --model best.pt

CLI: odp-infer --source 0 --model best.pt --threaded   # 摄像头实时
  │
  ├── cli/infer.py 解析参数 → InferService.predict(cli_args)
  │
  ├── 帧源识别:
  │   ├── str.isdigit() → CameraSource（设备号）
  │   ├── .mp4/.avi → VideoSource（视频文件）
  │   ├── .jpg/.png → ImageSource（单张图片）
  │   ├── 目录路径 → ImageFolderSource（文件夹）
  │   └── rtsp:// → CameraSource（RTSP 流）
  │
  ├── 模型加载:
  │   └── Detector(model_path) → YOLO(model_path)
  │       └── warmup() → GPU JIT 预热（仅 CUDA）
  │
  ├── 逐帧推理循环:
  │   ├── frame = source.read()
  │   ├── result = detector.detect(frame.image)
  │   │   ├── YOLO 前向传播 → NMS → Detection[]
  │   │   └── 每个 Detection: {class_id, class_name, confidence, bbox}
  │   ├── rendered = draw_detections(frame, detections)
  │   ├── rendered = draw_info_panel(rendered, fps, infer_ms, ...)
  │   └── 输出:
  │       ├── 写入 output.mp4（视频源）
  │       ├── 写入 frame_0000.jpg（图片源）
  │       └── cv2.imshow()（--show 参数）
  │
  └── 返回 InferResult:
      ├── stats = {frames, detections, per_class, avg_fps, avg_latency_ms}
      ├── output_dir = runs/infer/<experiment_name>/
      └── error（失败时）
```

**WebUI 单图检测流程**（前端→后端的完整链路）：

```
用户上传图片 → gr.Image(type="numpy")
  │
  ├── 点击"开始检测"
  │
  ├── Gradio 调用 _run_single_detection(image, model_path, conf, iou)
  │   │
  │   ├── _get_or_create_detector(model_path, conf, iou)
  │   │   ├── 检查 _detector_cache（dict + threading.Lock 线程安全）
  │   │   ├── 未命中 → Detector(model_path) → 存入缓存
  │   │   └── 命中 → 更新 conf/iou 参数
  │   │
  │   ├── detector.detect(image_np)
  │   │   ├── YOLO(image_np, conf=conf, iou=iou, verbose=False)
  │   │   ├── 解析 results.boxes → Detection[]
  │   │   └── 返回 InferenceResult
  │   │
  │   ├── draw_detections(image_np, detections)
  │   │   └── cv2.rectangle + cv2.putText → 标注图片
  │   │
  │   └── 返回 (标注图片, JSON检测明细, 状态文字, detector_state)
  │
  └── Gradio 渲染到 image_out + result_json + status
```

**WebUI 摄像头实时检测流程**：

```
用户点击"启动摄像头"
  │
  ├── _run_server_camera(camera_id, model_path, conf, iou)
  │   │
  │   ├── _release_server_camera() 释放前一个摄像头
  │   │
  │   ├── cap = cv2.VideoCapture(int(camera_id))
  │   │   ├── 后端: cv2.CAP_DSHOW（Windows）
  │   │   └── 设置分辨率: 640x480
  │   │
  │   ├── detector = _get_or_create_detector(model_path, conf, iou)
  │   │
  │   ├── 循环（while not _server_cam_stop.is_set()）:
  │   │   ├── ret, frame = cap.read()
  │   │   ├── result = detector.detect(frame_rgb)
  │   │   ├── rendered = draw_detections(frame, detections)
  │   │   └── yield rendered（生成器 → gr.Image 流式更新）
  │   │
  │   └── cap.release() + torch.cuda.empty_cache()
  │
  └── _server_cam_stop = threading.Event() 控制停止
```

**GPU JIT 预热机制**（`engine.py`）：

```python
def warmup(self, image_size=(640, 640)):
    """用纯黑图跑一次推理，消除首次 CUDA kernel 编译延迟。
    仅在 CUDA 设备上生效，CPU 推理跳过。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return
    dummy = np.zeros((*image_size, 3), dtype=np.uint8)
    self.detect(dummy)
```

### 2.8 webui/ — Gradio 前端

**路径**：`apps/platform/src/odp_platform/webui/`

**双模式设计**：

```
odp-webui 启动
  │
  ├── app.py → gr.Blocks(css=APP_CSS)
  │   │
  │   ├── 用户名密码登录（用户/管理员模式切换）
  │   │
  │   ├── 用户模式 Tab（6 个）:
  │   │   ├── 单图检测  ← create_single_detection_ui()
  │   │   ├── 文件夹检测  ← create_folder_detection_ui()
  │   │   ├── 视频检测  ← create_video_detection_ui()
  │   │   ├── 实时摄像头  ← create_live_camera_ui()
  │   │   ├── 模型选择  ← create_model_selection_ui()
  │   │   └── LLM对话  ← create_llm_chat_ui()
  │   │
  │   └── 管理员模式额外 Tab（+5 个）:
  │       ├── Dashboard ← create_dashboard_ui()
  │       ├── 模型演示 ← create_model_demo_ui()
  │       ├── 数据集浏览 ← create_dataset_browser_ui()
  │       ├── 训练 ← create_training_ui()
  │       ├── 数据校验 ← create_validation_ui()
  │       └── 配置管理 ← create_config_ui()
  │
  └── 启动 FastAPI 后端: subprocess(["uvicorn", "main:app", ...])
```

| 文件 | 职责 |
|------|------|
| `app.py` | Gradio 入口，组装配件 + CSS 样式 + 后端进程启动 |
| `user_tabs.py` | 6 个用户 Tab + 4 个检测函数 + Agent LLM 对话 |
| `llm_agent.py` | 关键词路由 Agent（模型/实验/推理/GPU 查询） |
| `training_tab.py` | 管理员训练 Tab（配置表单 + 训练启动 + 实时日志） |
| `dashboard.py` | 管理员 Dashboard |
| `utils.py` | 工具函数：`list_model_files()`（5 秒 TTL 缓存）、`run_python_module()`、JSON/CSV 导出 |
| `config_tab.py` | 配置管理 UI |
| `validation_tab.py` | 数据校验 UI |
| `model_demo.py` | 模型演示 Tab |
| `dataset_browser.py` | 数据集浏览 Tab |
| `dataset_analysis.py` | 数据集分析工具 |

**关键性能优化**：

```python
# 1. 模型缓存（避免切换 Tab 重复加载）
_detector_cache: dict[str, Detector] = {}
_cache_lock = threading.Lock()

# 2. 模型文件扫描缓存（5 秒 TTL）
@lru_cache(maxsize=1)
def list_model_files(ttl: int = 5) -> list[str]: ...

# 3. 摄像头资源管理
_server_cam_stop = threading.Event()
_server_cap_ref = [None]  # 单例引用，确保只有一个摄像头实例
```

### 2.9 cli/ — 命令行入口

**路径**：`apps/platform/src/odp_platform/cli/`

10 个命令，全部注册在 `pyproject.toml` 的 `[project.scripts]`：

| 命令 | 入口模块 | 调用链 |
|------|---------|--------|
| `odp-init` | `init_project.py` | → `paths.get_dirs_to_initialize()` → `Path.mkdir()` |
| `odp-reset` | `reset_project.py` | → 清理运行时目录 |
| `odp-transform` | `transform_data.py` | → `DatasetPipeline` → `service.converter_data_to_yolo()` → `core/*.py` |
| `odp-validate` | `validate_data.py` | → `validate_dataset()` → `run_all_checks()` → `checks/*.py` |
| `odp-config` | `config_cli.py` | → `run_config.service.build_config()` / `save_snapshot_to_file()` |
| `odp-train` | `train.py` | → `build_config()` → `run_experiment()` → `YOLO.train()` |
| `odp-val` | `val.py` | → `ValService.validate()` → `YOLO.val()` |
| `odp-infer` | `infer.py` | → `InferService.predict()` → `Detector.detect()` |
| `odp-webui` | `webui/app.py` | → `gr.Blocks().launch()` |
| `odp-backend` | `backend.py` | → `subprocess(["uvicorn", "main:app"])` |

### 2.10 web-backend/ — FastAPI 后端

**路径**：`apps/web-backend/`

**API 路由一览**：

| 路由 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/auth/register` | POST | 用户注册 |
| `/auth/login` | POST | 用户登录 |
| `/users/me` | GET | 当前用户信息 |
| `/users/me/history` | GET | 用户历史记录 |
| `/models` | GET/POST | 模型列表 / 上传模型 |
| `/models/upload` | POST | 模型文件上传 |
| `/models/{filename}` | DELETE | 删除模型 |
| `/experiments` | POST | 创建实验记录 |
| `/experiments/{id}` | PATCH | 更新实验状态 |
| `/experiments/{id}/epochs` | POST | 写入逐轮指标 |
| `/detection` | POST | 创建检测任务 |
| `/detection/{task_id}` | GET | 查询检测结果 |
| `/llm/chat` | POST | LLM 对话代理 |
| `/llm/models` | GET | 可用 LLM 模型列表 |
| `/validation/reports` | GET/POST | 质检报告 |

**后端不启动的影响**：

训练/Hooks 中对后端的调用都有 `try/except` + 超时 3 秒：
```python
try:
    requests.post(url, json=payload, timeout=3)
except requests.RequestException:
    logger.warning("后端不可达，实验仅保存在本地")
```

- 训练正常执行，不受影响
- 逐轮指标只写本地 CSV，不丢失
- Dashboard 只在后端启动时才有数据

---

## 3. 端到端全流程 (Architect 视角)

### 3.1 完整训练+推理链路

```
数据准备阶段:
  data/raw/rsod/              ← 原始数据存放位置
  ├── images/                   原始图片
  └── annotations/              Pascal VOC XML/COCO JSON/LabelMe JSON

数据转换:
  odp-transform --dataset rsod --format pascal_voc
  → configs/datasets/rsod.yaml    ← YOLO 训练配置文件
  → data/train/{images,labels}/  ← 转换后的 YOLO 格式训练集
  → data/val/{images,labels}/    ← 验证集
  → data/test/{images,labels}/   ← 测试集

数据校验:
  odp-validate --dataset rsod
  → yaml_schema: nc=5 ✓          ← 数据集配置正确
  → pair_existence: 100% ✓       ← 所有图片都有标签
  → label_format: 100% ✓         ← 标签格式正确
  → 报告: runs/data_validation/20260519_103000/

模型训练:
  odp-train --dataset rsod --model yolo11n.pt --epochs 100
  → runs/experiments/rsod_baseline/
      ├── config_snapshot.json    配置快照
      ├── results.csv             逐轮指标
      ├── weights/best.pt         最佳权重
      ├── results.png             训练曲线
      └── BoxPR_curve.png         PR 曲线
  → data/models/checkpoints/best_rsod_baseline.pt

模型评估:
  odp-val --model runs/experiments/rsod_baseline/weights/best.pt --dataset rsod
  → mAP50: 0.852, mAP50-95: 0.613, precision: 0.891, recall: 0.786

模型推理:
  odp-infer --model best.pt --source test.jpg
  → runs/infer/infer_20260519_best/
      ├── frame_000000.jpg       ← 标注图片
      └── audit.json             ← 推理记录
```

### 3.2 WebUI 全链路

```
用户浏览器                    Gradio 进程                Python 核心库
    │                            │                          │
    ├─ 打开 http://127.0.0.1:7860                          │
    │                            │                          │
    │                            ├─ 加载 CSS 样式            │
    │                            ├─ 创建 6 个用户 Tab       │
    │                            │                          │
    ├─ 上传图片 → 点击检测        │                          │
    │                            ├─ _run_single_detection() │
    │                            │                          ├─ Detector.detect()
    │                            │                          ├─ draw_detections()
    │                            │                          │
    ├─ ← 显示标注图片 + 检测列表                             │
    │                            │                          │
    ├─ 输入"有什么模型" → 发送     │                          │
    │                            ├─ run_agent()             │
    │                            │   └─ 关键词匹配"模型"     │
    │                            │                          ├─ tool_list_models()
    │                            │   └─ _format_with_llm()  │
    │                            │                          ├─ LLM API 调用
    │                            │                          │
    ├─ ← 显示"共 3 个模型: ..."                             │
    │                            │                          │
    ├─ 管理员登录 → 进入训练 Tab                              │
    │                            ├─ create_training_ui()    │
    │                            │   └─ 配置表单 → 启动训练  │
    │                            │                          ├─ run_experiment()
    │                            │                          │   ├─ YOLO.train()
    │                            │                          │   └─ TrainingHooks
    │                            │   └─ 实时日志流           │
    │                            │                          │
    ├─ ← 查看训练进度、结果曲线                               │
    │                            │                          │
    ├─ 点击"启动摄像头"                                      │
    │                            ├─ _run_server_camera()    │
    │                            │   ├─ cv2.VideoCapture()  │
    │                            │   ├─ Detector.detect()   │
    │                            │   ├─ draw_detections()   │
    │                            │   └─ yield → gr.Image    │
    │                            │                          │
    ├─ ← 实时查看摄像头检测流                                 │
```

---

## 4. Agent 智能助手系统

### 4.1 架构设计

`llm_agent.py` 实现关键词路由 + 本地工具执行 + LLM 排版：

```
用户消息 → Intent Router（正则匹配）
                │
    ┌───────────┼────────────┐
    │           │            │
  匹配工具    匹配推理     未匹配
    │           │            │
    ▼           ▼            ▼
 本地函数   提取路径     普通 LLM 对话
 执行工具   执行推理
    │           │
    └────┬──────┘
         ▼
  LLM 排版美化 → 自然语言回复
```

### 4.2 可用工具

| 关键词 | 执行函数 | 功能 |
|--------|---------|------|
| `模型\|model\|.pt\|权重` | `tool_list_models()` | 列出 `checkpoints/` 下所有 .pt 模型及大小 |
| `数据集\|dataset` | `tool_list_datasets()` | 列出所有数据集 YAML |
| `实验\|训练\|exp` | `tool_list_experiments()` | 列出实验 + best mAP50 |
| `推理\|检测\|识别` | `tool_run_inference()` | 自动提取模型+图片路径执行推理 |
| `GPU\|显存\|cuda` | `tool_get_gpu_info()` | PyTorch GPU 显存状态 |

### 4.3 为什么不用 LLM function calling？

`deepseek-v4-flash` 对 OpenAI 格式 function calling 支持不稳定（忽略 tools 参数）。关键词路由后：

- **100% 可靠**：工具执行不依赖 LLM
- **即时响应**：0 token 开销，本地直接执行
- LLM 只负责排版美化（一次 API 调用）

---

## 5. 模块依赖关系图

```
                           common/
                     (被所有模块依赖)
                           │
         ┌─────────────────┼──────────────────────┐
         │                 │                      │
    data_pipeline    data_validation          run_config
         │                 │                  (独立模块)
         └────────┬────────┘                      │
                  ▼                               │
             training ◄───────────────────────────┘
                  │
                  ▼
             evaluation
                  │
                  ▼
             inference ◄─── webui (Gradio)
                  │
                  ▼
            CLI 入口 (cli/)
                  │
                  ▼
           apps/web-backend/ (FastAPI, 可选)
```

**核心规则**：
1. `common/` 是纯基础库，不依赖任何业务模块
2. `run_config/` 独立于业务链，只被 CLI 和 training 调用
3. 业务模块单向依赖：data_pipeline → data_validation → training → evaluation → inference
4. `webui/` 直调 `inference/` 和 `training/`
5. `cli/` 是唯一入口，编排各业务模块

---

## 6. 答辩 FAQ / 常见架构问题

### Q1: 项目为什么是 Monorepo？

Monorepo = 多个子项目在同一个仓库里（ODPlatform 有 `apps/platform/` 和 `apps/web-backend/`）。

- 统一版本管理（一个 pyproject.toml 搞定 dev 工具）
- 跨模块修改原子化（一个 commit = 所有改动）
- 团队 < 5 人时，Monorepo 效率远高于多仓库

### Q2: `.odp-workspace` marker 怎么工作？

从 `__file__` 向上遍历父目录，找到包含该 marker 的目录即根目录。项目放任意路径都可用，无需硬编码。

### Q3: CLI 命令是怎么注册的？

`pyproject.toml` 的 `[project.scripts]` 段映射命令名到函数：
```
odp-train = "odp_platform.cli.train:main"
```
`pip install -e .` 后 pip 自动生成可执行脚本，每次执行 `odp-train` 实际调用 `train:main()`。

### Q4: WebUI 用户/管理员怎么切换？

用户模式 6 Tab（检测/摄像头/模型/LLM），管理员加 5 Tab（Dashboard/训练/校验/配置/模型演示/数据集浏览）。密码在 `app.py` 校验，无法直接 URL 访问。

### Q5: Agent 和 RAG 有什么区别？

RAG 查静态文档，Agent 执行实时代码。问"有什么模型"时 RAG 只能搜 README，Agent 直接扫描目录返回当前状态。

### Q6: 训练性能优化做了什么？

1. GPU JIT 预热（消除首次 CUDA 编译延迟）
2. WebUI 模型缓存（`_detector_cache`，切换 Tab 不重载）
3. EarlyStopping（patience=50，自动停止无效训练）
4. AMP 混合精度（默认开启，减少~40% 显存占用）
5. 模型列表 5 秒 TTL 缓存（`lru_cache`）

### Q7: CSV 列名适配器干什么的？

Ultralytics 升级时常改 CSV 列名（如 `metrics/mAP50(B)` → `mAP50`）。`_COLUMN_ALIASES` 映射表做兼容，升级 YOLO 只需加一行映射。

### Q8: 帧源注册表模式是什么？

`factory.py` 有个 `_SOURCE_REGISTRY` dict，`@register_source("name")` 装饰器自动注册。新增 RTSP 源只需 `@register_source("rtsp") class RTSPSource`，不用改 factory。

### Q9: 后端不可达会怎样？

不会。所有网络请求都有 3 秒超时 + try/except + 静默降级：
```python
try:
    requests.post(url, timeout=3)
except RequestException:
    logger.warning("后端不可达")
```
训练继续、指标写本地 CSV，后端恢复后可手动同步。

### Q10: CUDA OOM 怎么办？

1. 降 batch（最有效）
2. 降 imgsz（640→416）
3. 开 AMP（默认已开）
4. `device="cpu"` 回退 CPU
5. `torch.cuda.empty_cache()`

### Q11: 数据管道注册表怎么用？

```python
@register("my_format", supported_tasks=("detect",))
def convert_my_format(input_dir, output_labels_dir, options):
    # 实现转换逻辑
    return class_names  # 返回类别名列表
```
自动注册到 `_REGISTRY`，`service.py` 查询调用。新增格式=写一个文件+加个装饰器。

### Q12: CLI 和 WebUI 调用的是同一套代码吗？

是。`odp-train` CLI 和 WebUI 训练 Tab 最终都调 `training/experiment.py` 的 `run_experiment()`。推理也同理，`odp-infer` 和 WebUI 4 个检测 Tab 都调 `inference/engine.py` 的 `Detector.detect()`。

---

## 7. 学习路线图

### 7.1 按模块优先级

| 优先级 | 模块 | 核心文件 | 预计时间 |
|:------:|------|---------|:--------:|
| ⭐⭐⭐ | `common/` | paths.py, logging_utils.py | 2h |
| ⭐⭐⭐ | `data_pipeline/` | registry.py, orchestrator.py | 3h |
| ⭐⭐⭐ | `cli/` | 全部 10 个命令 | 2h |
| ⭐⭐ | `training/` | experiment.py, callbacks.py | 2h |
| ⭐⭐ | `inference/` | engine.py, frame_source/ | 2h |
| ⭐⭐ | `webui/` | app.py, user_tabs.py, utils.py | 2h |
| ⭐ | `run_config/` | service.py, merger.py | 1h |
| ⭐ | `data_validation/` | service.py, checks/*.py | 1h |
| ⭐ | `evaluation/` | service.py | 0.5h |

### 7.2 推荐阅读顺序

1. `common/paths.py` — 路径是一切的基础
2. `common/logging_utils.py` — 理解日志体系
3. `cli/transform_data.py` + `data_pipeline/` — 数据是训练的前提
4. `training/experiment.py` + `training/callbacks.py` — 训练核心
5. `apps/platform/src/odp_platform/inference/engine.py` + `inference/frame_source/` — 推理体系
6. `webui/app.py` + `webui/user_tabs.py` — 前端集成
7. `run_config/` — 配置管理
8. `data_validation/` — 质量保障

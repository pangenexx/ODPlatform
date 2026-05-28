# ODPlatform 架构师逐字讲解稿

> ⏱ 总时长：约 45-50 分钟 | 建议分两段讲（上半：架构，下半：演示+模式）
> 格式：`【动作】` = 动作提示 | **「演讲词」** = 逐字稿
> 建议：讲之前通读一遍，理解逻辑链条，不要照念

---

# 上半场：架构篇（约 25 分钟）

---

## 一、开场白（2 分钟）

【站定，微笑扫视听众，不要急着点PPT】

**「各位好，今天我来介绍一个平台，叫 ODPlatform。」**

**「先给大家看一句话——」**

【PPT 展示：一行大字】

> **一个让用户从原始标注数据到目标检测结果，全流程可视化的工程化平台。**

**「注意我这句话里没有出现任何一个技术名词——我没说 YOLO，没说 Gradio，没说 FastAPI。为什么？因为如果我们一上来就说 '这是一个基于 YOLOv11 用 Gradio 搭前端的检测平台'，听众的注意力就被分散到技术名词上了。他们会在想 'YOLO 是什么？Gradio 又是什么？'——而不是理解这个平台到底解决了什么问题。」**

**「所以我希望各位先记住这句话定义。当别人问你 '你们做了什么'，你不需要说技术栈，就说这一句话就够了。」**

**「今天我分三个部分来讲：」**

1. **「架构设计——为什么分五层，每层的职责是什么」**
2. **「功能演示——从 WebUI 的每个 Tab 到背后的调用链」**
3. **「核心模式——注册表、Marker 文件、三源合并，这些设计保证了扩展性」**

**「好，我们先从项目全景开始。」**

---

## 二、项目全景（3 分钟）

### 2.1 目录结构

【展示目录树 Slide】

**「我们先看看项目长什么样。」**

```
ODPlatform/
├── apps/platform/src/odp_platform/  ← 核心代码
│   ├── cli/             10 个命令入口
│   ├── common/          路径探测、日志系统、常量定义
│   ├── data_pipeline/   数据格式转换（VOC/COCO/YOLO）
│   ├── data_validation/ 数据集质量检查
│   ├── training/        训练引擎 + 回调系统
│   ├── inference/       推理引擎（Detector）+ 帧源（FrameSource）
│   ├── webui/           Gradio 前端
│   └── run_config/      配置管理（三源合并 + 溯源）
├── apps/web-backend/     FastAPI 后端（可选）
├── configs/datasets/     数据集 YAML 定义
├── data/runs/experiments/ 训练产物（权重/曲线/日志）
└── docs/                 架构决策记录 / 答辩问题集
```

**「这是一个标准的 monorepo——所有代码在同一个仓库里。你们看到这个 `apps/` 下面有两个子项目：`platform` 是核心引擎，`web-backend` 是一个可选的后端服务。」**

**「这里有一个细节值得注意：核心代码全在 `src/odp_platform/` 下面，而不是散落在各个目录。这意味着你 import 的时候永远是从 `odp_platform.xxx` 开始，路径清晰，不会出现 '这个模块到底在哪' 的困惑。」**

### 2.2 架构特点（四个核心）

**「在深入架构之前，我先说四个特点。这四个特点是整个项目的设计哲学——」**

**「第一，Monorepo。」**

**「什么是 monorepo？就是所有代码放在一个仓库里。为什么这么做？」**

**「假设你要改一个东西：CLI 加了一个参数，这个参数要传给 Service 层，Service 层要传给 Core 层。如果是多仓库，你需要：改 CLI 仓库 → 提 PR → 等合并 → 改 Service 仓库 → 提 PR → 等合并 → 改 Core 仓库 → 提 PR。三次 PR，三次 review，三次合并。」**

**「Monorepo 一次 PR 就搞定了。这就是最大的好处——跨模块变更的开销大大降低。」**

**「第二，Marker 文件。」**

**「大家写过项目都知道，最烦人的事情之一就是路径配置。`C:/Users/xxx/project/data/`——换个电脑就得改。更糟的是有人用相对路径，结果从不同目录启动就报错。」**

**「我们的做法是：在项目根目录放一个 `.odp-workspace` 文件。代码启动时，从当前文件的位置往上找这个文件。找到了，根目录就知道了。所有路径都基于这个根目录计算。」**

**「这意味着——项目你可以随便移动。换个文件夹、换个电脑、甚至换到 Linux——不需要改任何配置。」**

**「第三，注册表模式。」**

**「传统的 if/else 扩展方式：你要新加一个数据格式，就要打开格式分发函数，加一个 elif。假设有 10 种格式，那个函数就变成了 10 个 elif 的怪物——而且每次加的人都不一样，代码风格越来越乱。」**

**「注册表的做法：每个格式写一个单独的文件，在上面加一行 `@register` 装饰器。框架自动把格式名称和这个函数注册到一个字典里。新增格式只需要新建文件 + 加装饰器，不需要碰任何现有代码。」**

**「第四，三源合并。」**

**「一个配置项最终取什么值？可能来自 CLI 参数，可能来自 YAML 配置文件，可能来自代码的默认值。这三个来源的优先级是：CLI > YAML > 默认值。」**

**「而且我们不仅合并，我们还记录——每个字段最终从哪来的，都有 TraceRecord。你随时可以查到 '这个 epochs=200 是来自 CLI 还是来自 YAML'。」**

---

## 三、五层架构详解（12 分钟）

【走向白板，画出五层架构图，或者翻到架构图 Slide】

**「好，这是今天最核心的部分——五层架构。我会从下往上讲，因为理解顺序和依赖顺序是反的。」**

```
┌─────────────────────────────────────────┐
│            CLI 层（第 1 层）              │
│    10 个命令，只解析参数，不做业务          │
│    odp-train / odp-infer / odp-webui     │
└─────────────────┬───────────────────────┘
                  │ 调用
┌─────────────────▼───────────────────────┐
│          Service 层（第 2 层）            │
│    业务流程编排，一个函数一个完整用例       │
│    run_experiment / InferService         │
└─────────────────┬───────────────────────┘
                  │ 编排
┌─────────────────▼───────────────────────┐
│           Core 层（第 3 层）              │
│    单一业务逻辑，每个文件只做一件事         │
│    Detector / FrameSource / 格式转换器    │
└─────────────────┬───────────────────────┘
                  │ 读取
┌─────────────────▼───────────────────────┐
│          Config 层（第 4 层）             │
│    配置管理子系统，独立于业务链            │
│    三源合并 + 溯源 + 快照                 │
└─────────────────┬───────────────────────┘
                  │ 使用
┌─────────────────▼───────────────────────┐
│         Common 层（第 5 层）              │
│    基础设施，被所有模块依赖               │
│    paths / logging / constants           │
└─────────────────────────────────────────┘
```

### 3.1 Common 层——基础设施

**「先看最底层：Common 层。」**

**「这一层是基础设施，包含三个核心部分：」**

**「第一部分是路径系统——`paths.py`。核心逻辑就是刚才说的 Marker 文件探测。」**

```python
# common/paths.py
@cache
def _find_workspace_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / WORKSPACE_MARKER).exists():
            return parent
    raise FileNotFoundError(f"未找到 {WORKSPACE_MARKER}")

ROOT_DIR = _find_workspace_root(Path(__file__))
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = DATA_DIR / "models" / "checkpoints"
```

**「这个函数做了什么？从当前文件所在的位置开始，往父目录一层一层找，直到找到 `.odp-workspace` 文件。找到之后，那个目录就是项目根目录。」**

**「有一个 `@cache` 装饰器——这个函数只会执行一次，后续调用直接返回缓存结果，不重复遍历文件系统。」**

**「第二部分是日志系统——`logging_utils.py`。这个模块提供 `get_logger()` 函数，在程序入口调用一次，后续所有模块的 logger 自动继承。」**

```python
# 入口调用一次
get_logger(base_path=LOGGING_DIR, log_type="webui")

# 业务模块只需要一行
logger = logging.getLogger(__name__)
logger.info("训练开始")  # 自动冒泡到根 logger
```

**「设计上有个关键细节——幂等保护。如果 `get_logger` 被调了两次，第二次不会重复添加 handler，否则日志会打印两遍。」**

**「第三部分是常量定义——`constants.py`。所有的枚举、阈值、魔数都放在这里，而不是散落在各个业务模块中。」**

```python
FORMAT_CAPABILITIES = {
    AnnotationFormat.PASCAL_VOC: (Task.DETECT,),
    AnnotationFormat.COCO: (Task.DETECT, Task.SEGMENT),
    AnnotationFormat.YOLO: (Task.DETECT,),
}
```

**「Common 层最重要的原则是：它不依赖任何业务模块。如果哪天你发现 common 里的代码 import 了 training 或 inference 的东西——那就是架构违规，必须马上改。」**

### 3.2 Config 层——配置管理子系统

**「往上一层是 Config 层。这是一个独立的配置管理子系统，和具体的业务逻辑完全解耦。」**

**「先问一个问题：一个训练任务，参数可能来自哪些地方？」**

**「三个来源：」**

1. **「代码里的默认值——比如 `epochs=50`」**
2. **「YAML 配置文件——用户编辑了一个 `train.yaml` 文件，写了 `epochs: 100`」**
3. **「CLI 参数——用户在命令行输入了 `--epochs 200`」**

**「这三个来源的优先级是：CLI > YAML > 默认值。最终生效的是 `epochs=200`。」**

**「这就是三源合并的核心思想。我们来看具体实现——」**

```python
# run_config/merger.py
bundle = build_config(
    task="train",
    yaml_path=Path("train.yaml"),
    cli_args={"epochs": 200},
)
bundle.config   # 合并后的完整配置
bundle.trace    # 每个字段的来源链
```

**「`build_config` 函数做三件事：」**

1. **「加载 YAML 文件」**
2. **「合并 CLI 参数（覆盖 YAML 中同名字段）」**
3. **「对未设置的字段填充默认值」**

**「但真正有价值的是 `bundle.trace`——它记录了每个字段的来源链。」**

```python
# TraceRecord 示例
TraceRecord(
    field="epochs",
    final_value=200,
    sources=[
        ("代码默认值", 50),    # 初始值
        ("YAML 配置", 100),    # YAML 覆盖
        ("CLI 参数", 200),     # CLI 再覆盖
    ],
)
```

**「你可能会问：这个 trace 有什么用？两个场景：」**

**场景一：实验复现。「训练跑完了，结果很好。三个月后你想复现这个实验——但你已经不记得当时用了哪些参数了。没关系，Config 层每次训练都会保存一个 ConfigSnapshot。你只需要 `restore_from_snapshot()`，就能恢复当时的完整配置。」**

**场景二：排查问题。「队友跑的训练结果 MAP 特别低。你可以查配置快照，看是不是他用错了参数——比如 learning rate 设成了 0 ？」**

**「Config 层的文件结构是这样的——」**

```
run_config/
├── registry.py       @config_generator 装饰器 + 字段注册
├── service.py        build_config / restore_from_snapshot
├── loader.py         YAML 加载 + CLI 参数解析
├── merger.py         三源合并核心逻辑
├── schema.py         ConfigBundle / ConfigSnapshot / TraceRecord
├── validator.py      字段类型/范围/必填校验
└── fields/           各任务字段定义
    ├── train.py      训练相关字段
    ├── val.py        评估相关字段
    └── predict.py    推理相关字段
```

**「注意 `fields/` 目录——每个任务有自己的字段定义文件。同名字段在不同任务中可以有不同默认值。这是用 Pydantic 模型很难做到的——Pydantic 的字段是全局的，覆盖起来很麻烦。而我们用 ConfigField 注册表的方式，天然支持这种需求。」**

### 3.3 Core 层——业务逻辑实现

**「再往上一层是 Core 层。这一层做的是单一业务逻辑——每个文件只做一件事。」**

**「我们来看几个核心组件：」**

**第一个：Detector——推理引擎。」**

```python
# inference/engine.py
class Detector:
    def __init__(self, model_path, conf=0.25, iou=0.45):
        self._model = YOLO(model_path)

    def detect(self, image: np.ndarray) -> InferenceResult:
        results = self._model(image, conf=self.conf, iou=self.iou, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(Detection(...))
        return InferenceResult(detections=detections, ...)

    def warmup(self):
        """GPU JIT 预热——纯黑图跑一次，消除首次 CUDA 编译延迟"""
        if torch.cuda.is_available():
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.detect(dummy)
```

**「Detector 的职责非常清晰：加载模型，执行推理，返回结果。不做视频解码，不做图像保存，不做可视化。」**

**「`warmup()` 方法值得一提。深度学习模型第一次推理时，CUDA 会做 JIT 编译——这需要几秒钟。用户第一次点检测按钮就要等好几秒才有结果，体验很差。`warmup()` 在模型加载后用一张纯黑图跑一次推理——这次虽然也慢，但至少不是在用户操作的时候慢。后续推理就很快了。」**

**「第二个：FrameSource——输入源抽象。」**

```python
# frame_source/core/base.py
class FrameSource(ABC):
    @abstractmethod
    def open(self) -> bool
    @abstractmethod
    def read(self) -> Optional[Frame]
    @abstractmethod
    def close(self) -> None
```

**「FrameSource 是一个抽象基类，定义了三个核心方法：open、read、close。所有具体的输入源都继承这个基类。」**

**「我们有多个实现——」**

- **「CameraSource：读取摄像头」**
- **「VideoSource：读取视频文件」**
- **「ImageSource：读取单张图片」**
- **「ImageFolderSource：读取文件夹内的图片」**

**「每个实现只需要关心一件事：从哪里读取帧数据。不需要关心检测怎么做、结果怎么展示。」**

**「第三个：注册表模式下的数据转换器——」**

```python
# data_pipeline/core/pascal_voc.py
@register(AnnotationFormat.PASCAL_VOC, supported_tasks=(Task.DETECT,))
def convert_pascal_voc(input_dir, output_labels_dir, options):
    """Pascal VOC XML → YOLO txt。一个函数完成一个格式的转换。"""
    for xml_file in input_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        # 提取 bbox → 归一化 → 写 .txt
    return class_names
```

**「这个函数只做一件事：把 Pascal VOC 格式的 XML 标注文件转换成 YOLO 格式的 txt 文件。」**

**「新增一个格式——比如 LabelMe——只需要新建 `core/labelme.py`，加 `@register("labelme")` 装饰器。不需要改任何现有的代码。这就是开闭原则——对扩展开放，对修改关闭。」**

### 3.4 Service 层——业务流程编排

**「再往上是 Service 层。这一层做的是编排——把 Core 层的多个组件组合起来，完成一个完整的业务流程。」**

**「我们来看训练服务的例子——」**

```python
# training/experiment.py
def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # 1. 加载配置
    bundle = build_config(task="train", cli_args=config.to_dict())

    # 2. 初始化回调系统
    hooks = TrainingHooks(experiment_name=config.name, config_json=bundle.to_json())

    # 3. 启动训练
    model = YOLO(config.model)
    results = model.train(
        data=config.dataset_yaml,
        epochs=config.epochs,
        batch=config.batch,
        callbacks=hooks.get_ultralytics_callbacks(),
        **bundle.config,
    )

    # 4. 训练完成，归档模型
    shutil.copy("runs/train/exp/weights/best.pt", CHECKPOINTS_DIR / f"{config.name}.pt")

    # 5. 训练完成，保存快照
    bundle.save_snapshot(exp_dir / "config_snapshot.json")

    return ExperimentResult(best_map=results.map50, ...)
```

**「注意这个函数做了几件事：合并配置、初始化回调、启动训练、归档模型、保存快照。这些事如果放在 Core 层就太杂了——Core 层应该只做一件事。但放在 CLI 层也不行——CLI 层只应该解析参数。所以放在 Service 层是合适的——它编排了一个完整业务流程。」**

**「再来看推理服务——」**

```python
# inference/service.py
class InferService:
    def predict(self, cli_args: dict) -> InferResult:
        source_str = cli_args["source"]

        # 1. 创建帧源
        source = create_frame_source(source_str)

        # 2. 加载模型
        detector = Detector(cli_args["model"])

        # 3. 逐帧检测
        for frame in source:
            result = detector.detect(frame.image)
            annotated = draw_detections(frame.image, result.detections)
            # 保存或显示
            ...

        return InferResult(stats=...)
```

**「InferService 编排了帧源创建、模型加载、逐帧检测、结果保存这一整个流程。」**

**「说到这里，有一个诚实的话要说——我们的 WebUI 并没有经过 Service 层，而是直接调用了 Core 层的 Detector。按理说应该加一层 WebUI Service，但我们团队只有 5 个人，加了反而增加维护成本。这是务实的取舍。」**

### 3.5 CLI 层——命令入口

**「最顶层是 CLI 层。10 个命令，每个命令一个文件，每个文件一个 `_build_parser()` + 一个 `main()` 函数。」**

```python
# cli/train.py — 典型结构
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odp-train")
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    return parser

def main() -> int:
    args = _build_parser().parse_args()
    config = build_config(task="train", cli_args=vars(args))  # → Config 层
    result = run_experiment(config)                             # → Service 层
    print(f"mAP50: {result.map50:.4f}")
    return 0
```

**「注意 main() 函数只有三行：解析参数、调用 Service、打印结果。没有任何业务逻辑——业务逻辑在 Service 和 Core 层。」**

**「这些命令注册在 `pyproject.toml`：」**

```toml
[project.scripts]
odp-train = "odp_platform.cli.train:main"
odp-infer = "odp_platform.cli.infer:main"
odp-webui = "odp_platform.webui.app:main"
# ... 共 10 个
```

**「`pip install -e .` 之后，pip 会自动创建可执行脚本。所以你才能在终端直接输入 `odp-train` 来调用——不需要 `python -m xxx`。」**

---

## 四、数据流全链路演示（5 分钟）

**「前面讲了每一层的职责，现在我们把它们串起来——以一次训练任务为例，走一遍完整的数据流。」**

【PPT 展示数据流图】

**「用户在终端输入：」**

```bash
odp-train --dataset rsod --model yolo11n.pt --epochs 100
```

**「数据是这样流动的——」**

**「第一步：CLI 层」**

**「`cli/train.py` 的 `main()` 被调用。`argparse` 解析出三个参数：`dataset=rsod`、`model=yolo11n.pt`、`epochs=100`。然后调用 Service 层的 `run_experiment()`。到此，CLI 层的任务结束。」**

**「第二步：Service 层」**

**「`run_experiment()` 先调用 Config 层——把 CLI 参数传进去，同时读取默认的 train.yaml 配置文件。Config 层做三源合并：CLI 的 `epochs=100` 覆盖 YAML 的 `epochs=50`，YAML 的 `batch=16` 填充 CLI 没传的字段。」**

**「然后 Service 层加载模型——调用 Core 层的 `YOLO("yolo11n.pt")`。接着调用 `model.train()`——这是 Ultralytics 的库函数，不是我们写的。同时初始化回调系统——这些回调会在训练过程中把指标同步到 FastAPI 后端。」**

**「第三步：训练过程中」**

**「每个 epoch 结束，回调函数被触发。`TrainingHooks.on_epoch_end()` 把当前指标发给后端（如果后端在线）。如果后端不在线——HTTP 请求超时——回调会捕获异常，记一条 warning，然后继续训练。不中断。」**

**「第四步：训练完成」**

**「Service 层把 `best.pt` 复制到 `checkpoints/` 目录。调用 Config 层保存配置快照。打印最终 mAP。返回。」**

**「这条链路上，每一层只做自己的事：」**

- **「CLI：解析参数 → 传给 Service」**
- **「Service：编排流程 → 调用 Core + Config」**
- **「Core：单一逻辑 → 检测/训练」**
- **「Config：配置管理 → 合并/溯源/快照」**
- **「Common：基础设施 → 路径/日志」**

---

# 下半场：演示 + 模式篇（约 25 分钟）

---

## 五、CLI 命令详解（3 分钟）

【切换到终端，准备演示】

**「讲完了架构，我们来看具体怎么用。先快速过一遍 10 个命令——」**

| 命令 | 功能 | 背后调用 |
|------|------|---------|
| `odp-init` | 创建运行时目录 | `paths.get_dirs_to_initialize()` |
| `odp-reset` | 清理运行时数据 | `shutil.rmtree()` |
| `odp-transform` | 数据集格式转换 | `DatasetPipeline.pipeline()` |
| `odp-validate` | 数据集质检 | `validate_dataset()` |
| `odp-config` | 配置管理 | `build_config()` / `trace()` |
| `odp-train` | 训练 | `run_experiment()` |
| `odp-val` | 评估 | `ValService.validate()` |
| `odp-infer` | 推理 | `InferService.predict()` |
| `odp-webui` | 启动 Web 界面 | `gr.Blocks().launch()` |
| `odp-backend` | 启动后端服务 | `uvicorn.run()` |

**「注意一下：`odp-train` 这个 CLI 命令和后面要讲的 WebUI 管理员模式里的训练按钮——背后是同一个 `run_experiment()` 函数。不是两套实现。这一点在答辩时可能会被问到。」**

**「我们选两个命令演示一下——」**

```bash
# 查看训练命令有哪些参数
odp-train --help

# 启动 WebUI
odp-webui
```

---

## 六、WebUI 全功能演示（12 分钟）

【转向屏幕，打开浏览器，输入 http://127.0.0.1:7860】

**「好，接下来是重点——WebUI 功能演示。用户模式有 6 个 Tab，我逐个讲解。」**

### 6.1 单图检测

【点击"单图检测"Tab】

**「这个 Tab 的功能很直观：选模型 → 上传图片 → 点检测。」**

**「我们来演示一下——」**

【操作：选择模型、上传图片、点击检测】

**「点击检测后发生了什么？看代码——」**

```python
# WebUI 调用链
_run_single_detection(image, model_path, conf, iou)
  → _get_or_create_detector(model_path)  # 缓存查找/创建 Detector
  → detector.detect(image_np)            # Core 层推理
  → draw_detections(image_np, result)    # 画框
  → 返回 (标注图, JSON明细, 状态文本)
```

**「注意 `_get_or_create_detector()`——这是一个缓存函数。如果模型已经被加载过，就直接返回缓存中的 Detector 实例，避免重复加载。YOLO 模型几百 MB，每次操作都加载一次的话，用户会疯掉。」**

**「再看 JSON 明细——它列出了每个检测到的目标：类别、置信度、边界框坐标。这个 JSON 数据可以导出，用于后续分析。」**

### 6.2 文件夹检测

【点击"文件夹检测"Tab】

**「这个 Tab 支持两种输入方式：」**

1. **「点击上传多张图片文件——就像上传照片一样选文件」**
2. **「手动输入文件夹路径——适用于服务器上已有的数据集」**

**「另外还有一个输入框——最大处理张数。默认填 0 表示全部处理。如果你只想先看看效果，可以填 10——只处理前 10 张。」**

**「点击处理后的完整调用链：」**

```python
_run_folder_detection_wrapped(folder_path, model_path, conf, iou, max_images)
  → list_images(folder)         # 扫描文件夹，列出支持格式的图片
  → process_limit = min(max_images, total)  # 计算处理上限
  → for img in images[:limit]:  # 逐张处理
      detector.detect(img)
      draw_detections(img, result)
      → 结果存入 Gallery + 统计摘要
```

**「Gallery 展示所有检测后的图片，最多 500 张。底部有统计摘要——总图片数、成功数、总目标数、各类别分布。还有一个明细表格，列出每张图片的检测结果。」**

### 6.3 视频检测

【点击"视频检测"Tab】

**「上传一个视频文件，设置两个参数：」**

- **「跳帧间隔——每 N 帧检测一次。比如填 5，就是每 5 帧只检测 1 帧，其余 4 帧跳过。这是为了省时间——视频相邻帧之间变化很小，不需要逐帧检测。」**
- **「最大处理帧数——限制总处理量，防止视频太长等太久。」**

**「背后做的事情：」**

```python
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if not ret: break
    if frame_idx % frame_interval == 0:
        result = detector.detect(frame)
        rendered = draw_detections(frame, result)
        video_writer.write(rendered)     # 写入输出视频
    frame_idx += 1
```

**「检测完成后，你可以下载结果视频。同时也会展示一些样本帧 Gallery 和逐帧明细表格。」**

### 6.4 实时摄像头

【点击"实时摄像头"Tab】

**「这个 Tab 启动摄像头实时流，进行逐帧检测。」**

**「点击"启动"按钮后，后台启动了一个生成器函数——」**

```python
_run_server_camera(cam_id, model_path, conf, iou, cam_res)
  → 尝试打开摄像头（MSMF → DSHOW → 无后端降级）
  → while not stop_event:
      ret, frame = cap.read()
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      result = detector.detect(frame_rgb)
      rendered = draw_detections(frame_rgb, result)
      frame = draw_info_panel(rendered, fps, infer_ms, ...)
      yield frame  # → Gradio 流式更新
```

**「这里有几个关键设计：」**

**「首先是摄像头后端枚举。OpenCV 在 Windows 上有多个后端——MSMF、DSHOW。我们的代码按优先级尝试：先试 MSMF，不行就试 DSHOW，再不行就不指定后端让 OpenCV 自己选。这是为了兼容不同型号的摄像头。」**

**「其次是停止机制。点击"停止"按钮时，设置了一个 threading.Event，生成器检查到这个 Event 被设置了就退出循环，然后释放摄像头资源。」**

**「第三是信息面板——`draw_info_panel()` 在画面左上角叠加了 FPS、推理耗时、帧数、检测数等信息。这些信息对调试和演示都很有用。」**

**「注意：如果摄像头打不开——可能被其他程序占用——代码会进入 fallback 模式，显示一个"No Camera"的占位图，不会崩溃。」**

### 6.5 模型选择 + 实验训练结果

【点击"模型选择"Tab】

**「这个 Tab 做三件事：」**

1. **「选择当前要用的模型——从下拉框里选一个 .pt 文件」**
2. **「上传新的模型文件——或者手动输入模型路径」**
3. **「查看实验训练结果——这是比较有亮点的功能」**

【展开"实验训练结果"折叠面板】

**「训练结果展示区会显示以下内容：」**

```
┌─ 训练曲线 Tab ─────────────────────┐
│  Loss 曲线 + 验证指标曲线（动态绘制） │
│  最佳指标柱状图                      │
├─ 评估矩阵 Tab ─────────────────────┤
│  混淆矩阵 + 归一化混淆矩阵           │
│  PR 曲线 + F1 曲线                  │
├─ 类别分布 Tab ─────────────────────┤
│  类别分布直方图                      │
├─ 文本摘要 ─────────────────────────┤
│  实验名 / 轮数 / 最佳 mAP / 权重路径 │
└────────────────────────────────────┘
```

**「这些图表不是预先生成的图片——它们是从 `results.csv` 里动态绘制的。每次展开时，代码读取 CSV 数据，用 matplotlib 重新绘图。」**

**「为什么要动态绘制而不是直接展示预生成的图片？因为预生成的图片不一定存在——不同的 Ultralytics 版本生成的图片文件名可能不同。动态绘制从结构化的 CSV 数据生成，更可靠。」**

**「而且我们还做了列名适配——不同版本的 Ultralytics 可能用不同的列名，比如 `metrics/mAP50(B)` 和 `mAP50` 其实是同一个东西。`_COLUMN_ALIASES` 映射表解决了这个问题。」**

### 6.6 LLM 对话

【点击"LLM 对话"Tab】

**「最后一个用户 Tab 是 LLM 对话。接入的是 DeepSeek API。」**

**「填上 API Key，开启 Agent 工具——什么是 Agent 工具？当用户问 '有哪些模型可以用' 时，系统不是直接把问题扔给 LLM——而是先调用 `tool_list_models()` 函数去查真实的模型列表，然后把结果交给 LLM，让 LLM 用自然语言把结果组织成回复。」**

**「这样做的效果是：LLM 不会胡说八道——它查到的数据是真实的。它只是负责用自然语言把数据呈现出来。」**

### 6.7 管理员模式

**「点击右上角的齿轮按钮，输入密码 `0000`，进入管理员模式。」**

**「多了 5 个 Tab——」**

| Tab | 功能 |
|-----|------|
| **Dashboard** | 项目概览、后端连接状态 |
| **模型演示** | 加载模型 + 推理可视化，多模型对比 |
| **训练** | 配置训练参数 → 启动 → 实时看日志流 |
| **数据校验** | 运行质检 → 查看质检报告 |
| **配置管理** | 生成/验证/追踪配置模板 |

**「训练 Tab 值得一提——它的实时日志流功能。点击启动后，训练日志通过 generator 实时推送到前端，你可以看到每个 epoch 的 loss 变化，不用等训练结束了才知道结果。」**

---

## 七、核心设计模式详解（7 分钟）

【切回 Slide】

**「好，功能都看完了。我们回到代码层面——有哪些设计模式值得讲？」**

### 7.1 注册表模式

**「这是最核心的一个模式。它的核心思想是：把'有哪些东西'的登记工作和'怎么用这些东西'的调度工作分开。」**

**「没有注册表的时候，你是怎么加一个新格式的？」**

```python
# 传统 if/elif 方式
def convert(format_name, input_dir, output_dir):
    if format_name == "pascal_voc":
        ...  # 处理 VOC
    elif format_name == "coco":
        ...  # 处理 COCO
    elif format_name == "labelme":  # 新增：要改这个函数
        ...  # 处理 LabelMe
```

**「每加一个格式，你都要打开这个文件，加一个 elif。如果团队有 5 个人，各有各的格式要加——这个函数会变成什么样？」**

**「注册表的做法——」**

```python
# registry.py — 核心就是个字典
_REGISTRY: dict[str, Callable] = {}

def register(name: str):
    def decorator(func):
        _REGISTRY[name] = func
        return func
    return decorator

def get_converter(name: str) -> Callable:
    if name not in _REGISTRY:
        raise ValueError(f"未知格式: {name}")
    return _REGISTRY[name]
```

```python
# core/pascal_voc.py — 每个格式一个文件
@register("pascal_voc")
def convert_pascal_voc(input_dir, output_dir, options):
    ...
```

```python
# core/labelme.py — 新增格式：新建文件即可
@register("labelme")
def convert_labelme(input_dir, output_dir, options):
    ...
```

**「新增格式 = 新建一个文件 + 加一行 `@register`。不需要改 `registry.py`，不需要改 `orchestrator.py`，不需要改任何现有代码。」**

**「这里有一个值得对比的点——两种不同的调度方式：」**

**「**互斥分发 vs 聚合执行**」**

| | data_pipeline | data_validation |
|---|---|---|
| 调度方式 | **互斥分发**：按格式选 1 个 | **聚合执行**：全部 check 都跑 |
| 类比 | 点菜——选一道主食 | 体检——做全部项目 |
| 失败处理 | 选不到就报错 | 一个 ERROR 不影响其他 |
| 注册方式 | `@register("format_name")` | `@check("check_name")` |

**「前者用在数据转换——你不可能同时用 VOC 和 COCO 两种格式来处理同一份数据。后者用在数据质检——你需要跑所有的检查项，确保数据质量。」**

### 7.2 Marker 文件路径探测

**「这个模式解决的是一个非常实际的问题：项目路径定位。」**

**「很多项目是这样做的——」**

```python
# 不好的做法
ROOT_DIR = Path("C:/Users/xxx/projects/ODPlatform")  # 换电脑就废了

# 或者
ROOT_DIR = Path(os.getcwd())  # 依赖运行目录，从不同目录启动就错
```

**「我们的做法——」**

```python
# common/paths.py
WORKSPACE_MARKER = ".odp-workspace"

@cache
def _find_workspace_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / WORKSPACE_MARKER).exists():
            return parent
    raise FileNotFoundError(f"未找到 {WORKSPACE_MARKER}")

ROOT_DIR = _find_workspace_root(Path(__file__))
```

**「它的工作方式：从 `paths.py` 所在的位置开始，逐层向父目录找 `.odp-workspace` 这个文件。找到了就返回那个目录作为根目录。」**

**「好处是什么？项目你可以放在任何路径——`C:/project`、`D:/work/project`、`/home/user/project`——不需要改任何配置。你甚至可以把项目从 C 盘直接拖到 D 盘，照样能跑。」**

**「`@cache` 保证这个遍历只做一次，后续调用直接返回缓存结果。」**

### 7.3 三源合并 + 溯源

**「这个模式在 Config 层已经详细讲过。这里补充一个关键实现细节——」**

```python
# run_config/merger.py
def merge_config(defaults: dict, yaml_config: dict, cli_args: dict) -> ConfigBundle:
    trace_records = {}

    # 1. 从默认值开始
    merged = {}
    for key, value in defaults.items():
        merged[key] = value
        trace_records[key] = TraceRecord(
            field=key,
            final_value=value,
            sources=[("代码默认值", value)],
        )

    # 2. YAML 覆盖（如果字段在 YAML 中）
    for key, value in yaml_config.items():
        if key in merged:
            merged[key] = value
            trace_records[key].sources.append(("YAML 配置", value))

    # 3. CLI 再覆盖（如果字段在 CLI 参数中）
    for key, value in cli_args.items():
        if value is not None:  # CLI 没传的参数是 None，不覆盖
            merged[key] = value
            trace_records[key].sources.append(("CLI 参数", value))

    # 4. 更新最终值
    for key in trace_records:
        trace_records[key].final_value = merged[key]

    return ConfigBundle(config=merged, trace=TraceReport(records=trace_records))
```

**「这个函数的逻辑很清晰：先填默认值，再用 YAML 覆盖，最后用 CLI 覆盖。同时每一步都记录到 TraceRecord 中。」**

**「最终用户可以看到这样的溯源信息：」**

```
$ odp-config trace --field epochs
[epochs]
  代码默认值 → 50
  YAML 配置  → 100
  CLI 参数   → 200  ← 最终值
```

### 7.4 其他辅助技巧

**「还有几个辅助性的设计值得一提——」**

**「**GPU JIT 预热**——深度学习模型首次推理时 CUDA 会做 JIT 编译，耗时 3-5 秒。我们在模型加载后用纯黑图跑一次推理，把这个延迟提前消化掉。后续推理就很快了。」**

```python
def warmup(self):
    if torch.cuda.is_available():
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
```

**「**CSV 列名适配**——Ultralytics 不同版本的 results.csv 列名不一样。我们用别名映射自动兼容。」**

```python
_COLUMN_ALIASES = {
    "metrics/mAP50(B)": "map50",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}
```

**「**跨模块容错**——所有外部依赖（如 HTTP 请求）都有 try/except 兜底。」**

```python
try:
    requests.post(url, timeout=3)
except RequestException:
    logger.warning("后端不可达，仅保存在本地")
```

**「这意味着：后端挂了，训练照跑。类似这样的容错代码散布在整个项目中。」**

---

## 八、真实工程踩坑分享（3 分钟）

**「最后分享几个真实踩过的坑——这些都是我们写代码时撞上去的，代码里的注释直接写着的。」**

### 坑一：MSMF 帧率暴跌

**「在 Windows 上用 OpenCV 的 MSMF 后端读取摄像头，帧率只有 15fps——但换 DSHOW 后端就能到 30fps。查了两天发现是 MSMF 的硬件色彩转换滤镜导致的。」**

**「解决：在创建 VideoCapture 之前设置环境变量——」**

```python
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
```

**「注意这个环境变量必须在 `cv2.VideoCapture()` 之前设置才生效。这是我们在 camera.py 的注释里写着的——'撞墙记录'。」**

### 坑二：OpenCV 参数设置顺序

**「`cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)`——你以为设了就生效了？错了。」**

**「OpenCV 的 `set()` 方法只是 '请求' 一个参数，不是 '命令'。摄像头驱动可能不认。」**

**「而且设置顺序有要求——必须先设宽高，再设编码格式（FOURCC），最后设帧率（FPS）。顺序错了，分辨率不生效。」**

**「正确的做法——」**

```python
# 顺序：宽高 → FOURCC → FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, 30)

# 然后必须 read 一帧触发驱动协商
ret, _ = cap.read()

# 再用 get 读回真实值
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
```

### 坑三：CSV 列名变化

**「训练完成后画曲线，突然画不出来了。检查发现：Ultralytics 升级了一个小版本，`results.csv` 的列名从 `metrics/mAP50(B)` 变成了 `metrics/mAP50-95(B)`。」**

**「解决方案——别名映射：」**

```python
_COLUMN_ALIASES = {
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
}
# 读取 CSV 时，自动把旧列名映射到统一名称
```

### 坑四：日志重复输出

**「每次 import 一个模块，控制台就多打一遍日志。原因是 logger 的 propagate 机制——子 logger 的日志事件会冒泡到父 logger，如果父 logger 也配了 handler，就会打印两遍。」**

**「解决：」**

```python
# 设置 propagate=False 阻止冒泡
logger.propagate = False

# 同时加幂等保护，防止重复添加 handler
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
```

### 坑五：中文日志乱码

**「Windows 控制台显示 `???` 而不是中文。这是因为 Windows 默认编码是 GBK，而 Python 写的日志是 UTF-8。」**

**「解决：」**

```python
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
```

**「这些坑的价值在于——它们不是从书本上学来的，是在真实项目中撞上去的。如果有面试官或答辩老师问 '你们遇到过什么工程问题'，这些就是很好的素材。」**

---

## 九、总结与收尾（3 分钟）

**「好，以上就是 ODPlatform 的全部内容。我快速总结一下——」**

**「**架构**：五层架构——CLI 只解析参数，Service 编排流程，Core 实现单一逻辑，Config 管理配置，Common 提供基础设施。每一层职责清晰，依赖方向明确。」**

**「**核心模式**：注册表模式保证了扩展性——新增功能不碰旧代码。Marker 文件保证了可移植性——项目随便移动。三源合并保证了可复现性——每字段都可溯源。」**

**「**架构亮点**：」**
1. **「五层架构分层清晰，职责划分明确」**
2. **「注册表模式实现开闭原则，扩展性好」**
3. **「运行时不依赖后端，部署门槛低」**

**「**已知隐患**（诚实面对）：」**
1. **「training 模块的 Service 和 Core 合并在 `experiment.py` 中——应该拆开」**
2. **「WebUI 直接调 Core 层，跳过了 Service 层——应该加一层」**

**「这两个隐患我们清楚，但当前 5 人团队的规模下，它们是合理的取舍。未来随着团队扩大，有必要重构。」**

**「最后——一个问题留给大家思考：如果你的项目也需要支持多种输入格式、多种数据源，你会选择 if/elif 扩展还是注册表模式？两种选择的取舍是什么？」**

**「谢谢大家。下面留时间提问。」**

---

## 附录 A：应急话术速查

| 演示状况 | 话术 |
|---------|------|
| 摄像头打不开 | 「被其他程序占用了。我们看模型选择里的实验训练结果——那里有训练的完整数据，是更稳定的演示点」 |
| API Key 失效，LLM 没法用 | 「没关系，LLM 只是锦上添花。核心的检测和训练功能都不依赖它」 |
| 模型加载慢 | 「第一次 CUDA 编译，第二次就快了。正好说明我们的 GPU JIT 预热设计是必要的」 |
| 后端服务没启动 | 「后端是可选的，推理和训练完全不受影响——这是一个设计取舍」 |
| 代码演示出 bug | 「我们来看这个函数的容错设计——这正是工程化要考虑的边界情况」 |
| 听众问技术细节答不上 | 「好问题，我记一下，稍后确认后回复你」 |

## 附录 B：节奏控制

| 时段 | 内容 | 节奏 |
|------|------|------|
| 00:00-02:00 | 开场白 | 慢，定调 |
| 02:00-17:00 | 项目全景 + 五层架构 | 最慢，核心内容 |
| 17:00-22:00 | 数据流全链路 | 中速，把前面的串起来 |
| 22:00-25:00 | CLI 命令 | 快，过一遍就行 |
| 25:00-37:00 | WebUI 演示 | 中速，每个 Tab 演示 + 讲背后调用链 |
| 37:00-44:00 | 核心设计模式 | 中慢，重点讲注册表 |
| 44:00-47:00 | 踩坑分享 | 轻松，可以互动 |
| 47:00-50:00 | 总结收尾 | 慢，留提问时间 |

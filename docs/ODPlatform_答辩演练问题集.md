# ODPlatform 答辩演练问题集

> 面向项目架构师角色的答辩准备。问题来源于模拟答辩（grill-me）和框架补充，覆盖 D1~D5 全部文档、ADR 决策记录、团队协作方案。
>
> 每个问题标注了来源文档和考察意图，方便针对性复习。

---

## 目录

- [第一轮：全局架构（D1 + ADR-001）](#第一轮全局架构d1--adr-001)
- [第二轮：基础设施设计（D2）](#第二轮基础设施设计d2)
- [第三轮：数据流水线（D3 + ADR-002/005）](#第三轮数据流水线d3--adr-002005)
- [第四轮：数据质检（D4 + ADR-004/006）](#第四轮数据质检d4--adr-004006)
- [第五轮：运行配置（D5 + ADR-013）](#第五轮运行配置d5--adr-013)
- [第六轮：跨模块协同与接口契约](#第六轮跨模块协同与接口契约)
- [第七轮：Git 协作与工程规范](#第七轮git-协作与工程规范)
- [第八轮：框架综合与开放题](#第八轮框架综合与开放题)
- [附录：参考答案速查](#附录参考答案速查)

---

## 第一轮：全局架构（D1 + ADR-001）

### Q1-1 五层架构的数据流

你的架构是 `CLI → Service → Core → Config → Common` 五层。`odp-train` 这条命令从敲下去到真正开始训练，经过哪几层？每层干了什么？

> **来源**: D1-architecture.md 层架构图
> **考察**: 是否理解自己的架构分层，能否说清每层职责
> **提示**: CLI 层 `cli/train.py` 解析 argparse → Service 层 `service.py` 编排 → Core 层业务逻辑 → Config 层加载配置 → Common 层提供路径/日志工具

### Q1-2 Monorepo 与 Polyrepo 的选择

你选择了 Monorepo 而非 Polyrepo。举一个具体例子说明 Monorepo 的好处——比如某字段跨模块变更时的场景。

> **来源**: ADR-001-monorepo.md
> **考察**: 能否用具体场景而非抽象概念说明设计决策

### Q1-3 全局兜底的退出码

`odp-validate` 的退出码有 0/1/2/3。你的 `run_config`、`data_pipeline`、`training` 模块的退出码是否统一？谁定义的规则？跨模块调用中退出码怎么传递？

> **来源**: D4-需求规格说明书 2.4 节
> **考察**: 是否意识到退出码需要在全项目统一

### Q1-4 apps 的依赖关系

`apps/platform/` 和 `apps/web-backend/` 是一个整体还是两个独立服务？依赖方向是什么？如果后者挂了，前者还能工作吗？

> **来源**: ADR-001 + 团队协作指南"零"
> **考察**: 对 monorepo 内模块关系的理解

---

## 第二轮：基础设施设计（D2）

### Q2-1 marker file 定位根目录

`paths.py` 用 `.odp-workspace` marker file 向上查找根目录，为什么不用 `os.getcwd()` 或者 `__file__` 定位？如果我把 `apps/platform/` 移到别的位置，哪几个路径会坏？

> **来源**: D2-喂饭教程 阶段0.2、D2-任务说明"关键概念检查"
> **考察**: 理解"标记文件查找"模式的动机

### Q2-2 两层 pyproject.toml

你有两层 `pyproject.toml`——顶层和 `apps/platform/` 内层。各自放什么配置？为什么 `[project.scripts]` 要放在内层而不是顶层？换 hatchling 为 setuptools 要改哪几个地方？

> **来源**: D2-任务说明、D2-pyproject-toml 补充
> **考察**: 对 Python 包构建系统的基本理解

### Q2-3 日志 handler 配置

你规定所有人用 `get_logger()` 而不是 `print()`。`get_logger()` 的 handler 在哪里配的？配了几次？如果某人用 `logging.getLogger(__name__)` 而不是调 `get_logger()`，日志会去哪？

> **来源**: D2-任务说明"关键概念检查"第3题
> **考察**: 是否理解 logging 模块的 handler 机制

### Q2-4 `ROOT_DIR` 和 `APP_DIR` 的区别

`ROOT_DIR` 和 `APP_DIR` 的区别是什么？为什么 `data/` 放在 `ROOT_DIR`，`logging/` 放在 `APP_DIR`？

> **来源**: D2-任务说明"关键概念检查"第2题、paths.py
> **考察**: 对"项目根"vs"应用根"的理解——data 跨端共享，logging 是端私有

### Q2-5 entry-point 的两种调用方式

既然装包后能 `python -m odp_platform.cli.init_project`，为什么还要 `odp-init` 这个 entry-point？两者有什么区别？

> **来源**: D2-任务说明"关键概念检查"第7题
> **考察**: 理解 `[project.scripts]` 的本质——生成可执行脚本 vs `python -m` 调用

---

## 第三轮：数据流水线（D3 + ADR-002/005）

### Q3-1 注册表模式 vs 工厂模式

D3 用注册表模式替代 `if/elif` 分发转换器。为什么选了注册表而不是工厂模式或抽象基类？来一个 DOTA 格式的数据集，加一个 `core/dota.py`——需要改几个文件？

> **来源**: D3-讲义"阶段1：if/elif 漂移"、D3-任务说明"关键概念检查"第1题
> **考察**: 理解注册表模式的开闭原则优势——加 converter 只需新建文件 + 加装饰器

### Q3-2 覆盖率 fail-fast

orchestrator 设计了"覆盖率 fail-fast"——覆盖率低于 50% 的数据集拒绝转换。覆盖率怎么算的？放在哪个阶段执行的？为什么覆盖率检查要放在调 converter 之前？

> **来源**: D3-讲义"阶段9：覆盖率 fail-fast"
> **考察**: 理解"尽早失败"原则——如果先转换再检查，1000 张图 50 张标注会先跑完转换再报错，浪费时间

### Q3-3 COCO converter 用 tempfile

COCO converter 为什么要通过 `tempfile` 中转？不能直接在 `data/yolo/` 下写吗？如果不是 COCO 而是某种超大格式（如 100GB 的 HDF5），tempfile 模式还适用吗？

> **来源**: D3-讲义"阶段3：COCO converter + tempfile 中转"
> **考察**: 理解 tempfile 的隔离性——防止转换失败后残留脏数据

### Q3-4 YOLO 格式的"接口等价性"

YOLO 已经是目标格式了，为什么它的 converter 还要"写"到 `output_labels_dir`？"接口等价性"解决了什么问题？

> **来源**: D3-讲义"阶段4：YOLO converter + 接口等价性"
> **考察**: 理解接口等价性的价值——上层 orchestrator 不需要区分"这是转换"还是"这是拷贝"

### Q3-5 数据集目录约定

你规定 `data/raw/<数据集名>/{images, annotations}/`，严令禁止 `data/raw/temp_xxx/` 这种目录出现。这条铁律的目的是什么？"消除歧义"具体指什么？

> **来源**: D3-讲义"阶段0.1 + 0.2"
> **考察**: 是否理解数据目录约定在大型项目中的作用

---

## 第四轮：数据质检（D4 + ADR-004/006）

### Q4-1 DatasetSnapshot 的设计

`DatasetSnapshot` 为什么设计成 `frozen=True` + 内部容器全是 `Tuple`？如果某个 check 试图 `snapshot.images_per_split['train'].append(...)`，会怎么样？

> **来源**: D4-需求规格说明书 3.1 节 DatasetSnapshot 铁律
> **考察**: 是否理解 frozen + Tuple 的双层不可变设计

### Q4-2 CheckResult 的 severity 语义

`CheckResult` 的 `summary` 和 `details` 有什么区别？为什么 `INFO` 也算 `passed = True`？如果所有 check 都 PASS 但有一个 INFO，退出码应该是多少？

> **来源**: D4-需求规格说明书 3.1 节
> **考察**: 理解 PASS/INFO/WARNING/ERROR 四级语义

### Q4-3 四个 check 的执行顺序

你有 4 个 check——yaml_schema、pair_existence、label_format、split_uniqueness。如果 yaml_schema ERROR，label_format 也 ERROR，pair_existence WARNING——退出码是多少？用户该先改哪个？

> **来源**: D4-讲义"阶段1-4"
> **考察**: 是否理解 check 间的隐式依赖——yaml_schema 是前置，修完之后 label_format 可能自己就好了

### Q4-4 聚合执行 vs 互斥分发

D3 data_pipeline 是"互斥分发"（按格式选 1 个 converter），D4 data_validation 是"聚合执行"（跑全部 check）。两种调度模式的区别是什么？为什么 D4 不能像 D3 一样一个报错就停？

> **来源**: D4-讲义"D4 跟 D3 的关系"表格
> **考察**: 理解"产线"和"质检"不同的失败语义

### Q4-5 质检只检测不修复

你的设计原则是"质检子系统只检测，不修复"。如果发现问题，谁负责修？如果混在一起会有什么问题？

> **来源**: D4-讲义 设计原则
> **考察**: 理解关注点分离——validate vs repair 混在一起会导致"跑了一次 validate 数据是否变了"说不清

---

## 第五轮：运行配置（D5 + ADR-013）

### Q5-1 ConfigField vs Pydantic 模型

你选择了"ConfigField 注册表"而不是"Pydantic 模型继承"做配置方案。为什么？"字段为中心而非任务为中心"怎么理解？

> **来源**: ADR-013-run-config.md "背景 + 选项"
> **考察**: 理解配置系统的核心设计决策

### Q5-2 三源合并的覆盖链

"三源合并"（代码默认值 → YAML → CLI）中，每个字段的"来源链"是怎么记录的？用一个具体例子说明：某个字段在代码里默认 0.01，YAML 里写了 0.02，CLI 传了 0.005——最终值是多少？来源链是什么样的？

> **来源**: ADR-013-run-config.md "选项B" + D5-SRS FR-10/FR-11
> **考察**: 理解 TraceRecord 的结构化溯源能力

### Q5-3 未知字段检测

SRS FR-14 要求"YAML 中字段名拼写错误时立即报错，不能静默忽略"。run_config 是怎么检测未知字段的？用户在 YAML 里写了 `epchs: 200`（应该是 `epochs`）——系统在哪个阶段报错？

> **来源**: D5-SRS FR-14、AC-06
> **考察**: 理解 loader 阶段的字段名白名单校验机制

### Q5-4 ConfigSnapshot 的作用

`ConfigSnapshot` 包含哪些字段？为什么不用 Pydantic 模型序列化的结果作为快照？快照能恢复出配置吗？

> **来源**: ADR-013-run-config.md "选项B ConfigSnapshot"
> **考察**: 理解快照的"纯净性"——只存 `{field: value}` 字典，不携带类结构信息

### Q5-5 敏感字段脱敏

FR-24 要求敏感字段脱敏。ConfigField 怎么标记一个字段是敏感的？脱敏在哪个阶段进行？如果用户想查看已脱敏的字段怎么做？

> **来源**: D5-SRS FR-24、ADR-013-run-config.md "敏感字段脱敏"
> **考察**: 理解 `sensitive=True` 标记 + `to_report_dict(mask_sensitive=True)` 机制

### Q5-6 注册表的幂等设计

`register_field()` 被设计为幂等的——同名首次注册胜出，后续静默跳过。这个设计的动机是什么？如果 predict.py 和 train.py 都想注册 `device` 字段，谁的版本会被使用？

> **来源**: ADR-013-run-config.md "register_field 幂等设计"
> **考察**: 理解"首次胜出"策略——`__init__.py` 中通用定义优先，任务特有定义的 device 被安全忽略

---

## 第六轮：跨模块协同与接口契约

### Q6-1 训练→后端的容错

训练→后端的接口有 `on_epoch_end(exp_id, epoch, metrics)`。训练 300 epoch，每个 epoch 5 秒，后端在第 150 epoch 宕机。剩下 150 epoch 的数据会怎样？3 次重试用完后训练还在继续吗？后端恢复后缺失的 epoch 怎么补？

> **来源**: 团队协作指南 5.2、callbacks.py TrainingHooks
> **考察**: 理解"训练不依赖后端"的设计原则——重试耗尽只丢数据，不中断训练

### Q6-2 算法工程师的实验入口

算法工程师要做消融实验，你规定他不能直接调 ultralytics，必须通过 `run_experiment()`。他怎么改损失函数？怎么改 NMS？怎么加 CBAM 模块？你的入口文件考虑过这些扩展需求吗？

> **来源**: 团队协作指南 5.5、6.2
> **考察**: 理解 `ExperimentConfig(model="...")` 传自定义 yaml 的扩展机制

### Q6-3 前端对推理模块的依赖

前端的 `model_demo.py` 调用推理模块的 `Detector`。如果用户选了模型、点了推理，但推理模块还没写好——前端会怎样？你的方案里怎么处理这种依赖关系？

> **来源**: 团队协作指南 5.4、6.3 model_demo.py
> **考察**: 理解 `gr.State(None)` + `interactive=False` 的前置校验

### Q6-4 数据结构变更的影响面

推理模块的 `Detection.bbox` 如果从 `(x1,y1,x2,y2)` 改成 `(cx,cy,w,h)`——谁会受影响？你怎么保证这个变更不会静默出错？

> **来源**: 团队协作指南 5.4、inference/engine.py + visualizer.py + cli/infer.py
> **考察**: 是否有"调用链分析"思维——三个消费者全部受影响

### Q6-5 训练→推理的模型文件传递

训练产出的 `best.pt` 怎么传给推理模块？文件名怎么保证不冲突？如果两个实验都产出 `best.pt` 但指向不同实验目录——推理模块怎么知道该读哪一个？

> **来源**: 团队协作指南 5.1、6.1
> **考察**: 理解 `best_{config.name}.pt` 的命名约定 + `ExperimentResult.model_path` 的绝对路径传递

### Q6-6 算法实验的运行规范

算法工程师做消融实验必须 `seed=42`、单变量改动。为什么 seed 要固定？为什么每次只能改一个变量？

> **来源**: 团队协作指南 6.2
> **考察**: 理解消融实验的基本方法论

### Q6-7 跨模块容错规范

你的协作指南里规定了 6 条跨模块容错（A~F）。选其中一条说明：如果不遵守会有什么后果？给出非培训师写的代码反例。

> **来源**: 团队协作指南 6.6
> **考察**: 是否理解每条容错规范的实际意义

---

## 第七轮：Git 协作与工程规范

### Q7-1 分支策略

你用了 `feat/training`、`feat/algorithm` 等 5 个 feature 分支 + main（保护）。如果训练工程师的代码依赖推理工程师刚写好的 `Detector`，但推理的 PR 还没合并——训练怎么推进？有没有方案避免阻塞？

> **来源**: 团队协作指南"八、Git 分支与阶段"
> **考察**: 理解 feature 分支的依赖管理——松耦合设计 + 先合并不依赖其他模块的部分

### Q7-2 写权限矩阵

你的写权限矩阵规定了每个人的目录边界。如果前端工程师发现 `inference/visualizer.py` 的颜色映射不够清晰顺手改了——你 approve 还是打回去？严格的边界会不会拖慢开发？

> **来源**: 团队协作指南 3.2 写权限矩阵 + 七、协同检查清单
> **考察**: 理解"默认规则 + 灵活例外"的小团队合作哲学

### Q7-3 提交范围

你给每个人规定了提交范围。如果一次 PR 无意间跨越了目录边界（比如算法工程师修了 `experiment.py` 里的 bug）——怎么处理？`git add -A` 有什么风险？

> **来源**: 团队协作指南每个角色最后的"提交范围"段
> **考察**: 是否理解精确的 `git add <文件>` 而非 `git add -A` 的重要性

### Q7-4 协同检查清单

你的协同检查清单有 13 条。哪一条是你觉得最容易被忽略的？怎么在 PR review 中自动/半自动检查它？

> **来源**: 团队协作指南"七、协同检查清单"
> **考察**: 对协同检查清单的实际使用体验

### Q7-5 测试分层策略

ADR-011 定义了测试分层策略。你的项目中单元测试、集成测试、端到端测试各覆盖什么？验收脚本只测试"成功场景"，失败场景怎么办？

> **来源**: ADR-011-test-strategy.md（ADR 集合中）
> **考察**: 理解测试金字塔 + 失败场景的测试覆盖

---

## 第八轮：框架综合与开放题

### Q8-1 全流程追踪

从"用户在前端填写训练参数并点击开始训练"到"用户在前端看到检测结果"——完整走一遍你架构里的数据流。每个模块的输入输出是什么？不准确模糊的词汇，用文件名和函数名说话。

> **来源**: 全文档综合
> **考察**: 对项目整体数据流的掌握程度

### Q8-2 支持视频流实时检测

如果明天要让 ODPlatform 支持视频流实时检测（RTSP 推流 → GPU 推理 → Web 页面实时显示），你的架构里哪几层需要改？哪几层不用改？

> **来源**: ADR-008 推理流水线异步化、ADR-009 frame_source 输入源抽象
> **考察**: 架构的可扩展性——加新输入源不改推理逻辑

### Q8-3 易扩展性

你的架构里哪个子系统最容易扩展？哪个最难扩展？为什么？

> **来源**: 全文档综合
> **考察**: 对架构各子系统的扩展成本判断

### Q8-4 最大的隐患

你觉得你这个架构最大的亮点是什么？同时最大的隐患是什么？两个都要答，不能只说好的。

> **来源**: 全文档综合
> **考察**: 架构师视角——既知道自己的优势，也能诚实面对缺陷

### Q8-5 一句话定义项目

用一句话（不超过 30 字）说明这个项目解决了什么核心问题——不准提具体技术名词（YOLO、Gradio、FastAPI 都不准出现）。

> **来源**: 答辩开场白和结束语
> **考察**: 能否用非技术语言讲清项目价值

### Q8-6 跨模块字段变更流程

训练工程师想在 `ExperimentConfig` 里新增一个 `use_wandb: bool = False` 字段。从修改到上线，涉及几个角色、几个文件、几次 PR？

> **来源**: 团队协作指南 2.1 + 2.3 + 七
> **考察**: 对"参数源头 → 后端同步 → 前端同步"全链路的理解

### Q8-7 数据安全管理

`data/` 被 `.gitignore` 排除了。新成员怎么拿到 RSOD 和 VisDrone 数据集？怎么保证不同成员的数据集版本一致？

> **来源**: 团队协作指南"零"
> **考察**: 对数据分发的安全性 + 一致性的考虑

### Q8-8 实验可复现性

`ExperimentConfig` 的 `seed=42`、`ConfigSnapshot`、`run_id`——你的架构从哪些维度保证实验可复现？如果有人跑了一个实验但没保存 log，还能复现吗？

> **来源**: ADR-013 ConfigSnapshot、D5-SRS FR-23
> **考察**: 理解"可复现"不仅是种子固定

---

## 附录：参考答案速查

### Architecture ADR 索引（快速定位）

| 问题编号 | 核心 ADR | 关键代码文件 |
|---|---|---|
| Q1-1, Q1-4 | ADR-001 | `cli/train.py`, `common/paths.py` |
| Q2-1 ~ Q2-5 | ADR-002/003/010 | `common/paths.py`, `logging_utils.py`, `pyproject.toml` |
| Q3-1 ~ Q3-5 | ADR-002/005 | `data_pipeline/registry.py`, `orchestrator.py` |
| Q4-1 ~ Q4-5 | ADR-004/006 | `data_validation/registry.py`, `snapshot.py` |
| Q5-1 ~ Q5-6 | ADR-013 | `run_config/fields/`, `merger.py`, `service.py` |
| Q6-1 ~ Q6-7 | — | `training/callbacks.py`, `inference/engine.py`, `webui/model_demo.py` |
| Q7-1 ~ Q7-5 | ADR-011 | `.github/workflows/`, `pyproject.toml` |
| Q8-1 ~ Q8-8 | ADR-007/008/009 | 全部 |

### 高频考点 Top 5（优先准备）

1. **D5 run_config 三源合并 + 溯源**（Q5-1/Q5-2/Q5-3）——最薄弱且最高频
2. **全流程数据追踪**（Q8-1）——决定整体印象
3. **跨模块接口容错**（Q6-1/Q6-3）——体现工程化思维
4. **Monorepo 优劣 + 具体例子**（Q1-2）——基础必考
5. **D3 注册表 vs D4 注册表的调度模式区别**（Q4-4）——体现架构模式复用的理解
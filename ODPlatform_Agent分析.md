# ODPlatform Agent 技术分析

> 分析对象：`apps/platform/src/odp_platform/webui/llm_agent.py`
> 集成入口：`apps/platform/src/odp_platform/webui/user_tabs.py` 中 `_chat()` 函数

---

## 一、Agent 是什么

ODPlatform 的 Agent 是一个**关键词驱动的工具调用系统**，集成在 WebUI 的"LLM对话"Tab 中。用户输入自然语言 → 系统识别意图 → 执行本地工具 → LLM 美化输出。

它不是 LLM Function Calling（如 OpenAI 的 `tools` 参数）——不依赖 LLM 自己判断调哪个工具。而是**本地正则匹配 + 路由表**，确定调什么工具，调完后把结果塞给 LLM 让它写人话。

---

## 二、能干什么

Agent 目前注册了 **5 个工具**，一个带参数执行推理，4 个无参数查询：

### 2.1 `tool_list_models()` — 列出可用模型

扫描 CHECKPOINTS_DIR 和 models/checkpoints 下的所有 .pt 文件，返回列表含文件大小。

```
[INFO] 共 3 个模型:
  - F:\ODPlatform\data\models\checkpoints\best_rsod.pt (42.3MB)
  - F:\ODPlatform\data\models\checkpoints\yolo11n.pt (5.3MB)
```

触发关键词：`模型` / `model` / `.pt` / `权重` / `checkpoint`

### 2.2 `tool_list_datasets()` — 列出数据集

扫描 CONFIGS_DATASETS_DIR 下所有 .yaml 配置文件，返回数据集名称列表。

触发关键词：`数据集` / `dataset`

### 2.3 `tool_list_experiments()` — 列出实验及最佳 mAP

遍历 RUNS_DIR/experiments/ 下每个子目录，读取 results.csv 找出最优 mAP50。

触发关键词：`实验` / `训练` / `训练结果`

### 2.4 `tool_get_experiment(name)` — 查单个实验详情

读取指定实验的 results.csv 和 config_snapshot.json，返回 JSON（含最终指标、最佳轮次、完整配置）。

### 2.5 `tool_run_inference(model, image, conf, iou)` — 执行推理

加载 Detector → warmup → 读图 → detect → 返回检测结果列表。这是唯一一个有副作用的工具——真正跑模型推理。

触发关键词：`推理` / `检测` / `识别` / `infer` / `detect`

### 2.6 `tool_get_gpu_info()` — 查询 GPU 状态

通过 PyTorch 的 CUDA API 获取显存使用情况。

触发关键词：`GPU` / `显存` / `显卡` / `cuda`

---

## 三、架构与数据流

### 3.1 整体架构

```
用户输入
    │
    ▼
┌────────────────────────────────────────┐
│          Gradio Chat UI                │
│  user_tabs.py → _chat()               │
│    ├── enable_tools=False → _simple_chat()  │
│    └── enable_tools=True  → run_agent()     │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│          llm_agent.py                  │
│                                        │
│  run_agent()                           │
│    ├── Step 1: 推理关键词匹配            │
│    │    (检测/推理 → resolve路径 →      │
│    │     tool_run_inference)           │
│    ├── Step 2: 通用工具路由             │
│    │    (遍历 _TOOL_MAP 正则匹配表)      │
│    ├── Step 3: 匹配成功 → LLM 美化      │
│    │    (_format_with_llm)             │
│    └── Step 4: 无匹配 → 普通聊天        │
│         (_simple_chat_fallback)        │
└────────────────┬───────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 工具函数     DeepSeek API   Core 层
 (5个)       (/chat/completions)  (Detector)
```

### 3.2 完整数据流

以用户输入"帮我检测图片 F:/test.jpg 用 best.pt 模型"为例：

```
Step 1: _chat() 收到消息
  │
Step 2: enable_tools=True → run_agent()
  │
Step 3: re.search(r"推理|检测|infer|detect", text) 匹配成功
  │
Step 4: _resolve_model_path(text)
  │  ├── 正则提取 "F:/test.jpg" → 失败（不是 .pt）
  │  └── 尝试模型名模糊匹配 → "best" 匹配到 "best_rsod.pt"
  │
Step 5: _resolve_image_path(text)
  │  └── 正则提取 "F:/test.jpg" → 成功
  │
Step 6: tool_run_inference("F:/.../best_rsod.pt", "F:/test.jpg")
  │  ├── Detector(str(model_path))        ← Core 层
  │  ├── detector.warmup()                ← GPU JIT 预热
  │  ├── cv2.imread → cv2.cvtColor        ← OpenCV
  │  ├── detector.detect(img_rgb)         ← YOLO 推理
  │  └── 返回格式化文本结果
  │
Step 7: _format_with_llm(user_text, tool_result, api_key, ...)
  │  ├── System: "把工具结果用自然语言回答"
  │  ├── User: "用户问题: ...\n\n工具返回结果:\n[OK] 推理完成\n..."
  │  └── → DeepSeek API → "已用 best_rsod.pt 模型完成检测，共识别到 3 个目标..."
  │
Step 8: history.append({"role": "assistant", "content": formatted})
  │
Step 9: 返回 (history, "") → Gradio 刷新聊天界面
```

### 3.3 关键词路由表机制

```python
_TOOL_MAP: list[tuple[re.Pattern, str, Callable, list[str]]] = []

def _route(pattern: str, tool_name: str, fn: Callable, params: list[str] | None = None):
    _TOOL_MAP.append((re.compile(pattern, re.IGNORECASE), tool_name, fn, params or []))

_route(r"模型|model|\.pt|权重|checkpoint", "list_models", tool_list_models)
_route(r"数据集|dataset|数据", "list_datasets", tool_list_datasets)
_route(r"实验|训练|exp|train.*结果|训练结果", "list_experiments", tool_list_experiments)
_route(r"GPU|显存|显卡|cuda|gpu", "tool_get_gpu_info", tool_get_gpu_info)
```

路由逻辑：
1. 首先走**推理专用分支**（硬编码关键词匹配）
2. 没匹配到推理，再遍历 `_TOOL_MAP` 顺序匹配
3. 先注册的优先匹配
4. 匹配到第一个就返回，不会多个工具同时执行

---

## 四、技术优势

### 4.1 不依赖 LLM Function Calling

**这是最大的特点。** OpenAI 的 function calling、DeepSeek 的 tool call——这些需要 LLM 自己决定调用哪个工具。问题是：
- 小模型 function calling 不稳定，可能不调用/乱调用
- 每次请求都带 tools 描述，消耗上下文长度
- LLM 返回的 tool call 参数可能格式不对

ODPlatform 的做法：**正则匹配比 LLM 稳定得多**。用户说"看看有什么模型"，正则 100% 匹配到 `list_models`，不需要 LLM 参与决策。

### 4.2 离线执行

工具函数全部在本地运行，不依赖网络（除了最终 LLM 美化那一步）。即使 DeepSeek API 挂了，工具仍然能执行——只是无法用自然语言回复。

### 4.3 可审计

工具返回的结果是结构化的、确定的。不存在 LLM 编造数据的风险（hallucination）。LLM 只负责"把数据润色成自然语言"，不负责"判断数据是什么"。

### 4.4 容错设计

每个工具函数内部都有 try/except 保护：
- 文件不存在 → 返回 `[ERROR]` 消息
- 导入失败 → 返回 `导入失败`
- 推理异常 → 返回 `推理失败: {异常信息}`

Agent 层也有 try/except：
```python
try:
    from odp_platform.webui.llm_agent import run_agent
    return run_agent(...)
except ImportError as exc:
    history.append({"role": "assistant", "content": f"Agent 模块未就绪: {exc}"})
```

### 4.5 路径参数智能解析

`_resolve_model_path()` 做了两件事：先尝试正则提取 .pt 路径，如果没找到就尝试模型名模糊匹配。用户说"用 best 模型"，工具能自动映射到 `best_rsod.pt`。这个模糊匹配在真实使用场景中很实用。

### 4.6 可开关的 Agent 模式

UI 上有一个 `enable_tools` Checkbox，关掉就退化为纯 LLM 聊天。用户可以在"工具模式"和"闲聊模式"之间切换。

---

## 五、不足与改进方向

### 5.1 关键词匹配的局限性（最大不足）

| 问题 | 例子 | 后果 |
|------|------|------|
| 语义盲区 | "给我展示下训练跑出来的结果" → 匹配不到"实验"或"训练结果" | 走 fallback 纯 LLM 聊天 |
| 误匹配 | "这个模型检测效果怎么样" → 匹配到"检测" → 触发推理工具 | 用户可能只是想问模型效果，而不是实际跑推理 |
| 单意图 | 一句话只能触发一个工具 | 无法同时查模型和数据集 |
| 无上下文 | "那个实验呢？" → 无法跟踪"那个"指代哪个 | 纯正则无法做指代消解 |

**正则匹配的准确率取决于关键词覆盖的完整度，本质上是个需要持续维护的黑名单/白名单。**

### 5.2 推理工具的路径依赖

`tool_run_inference` 要求用户输入显式的文件路径。但在 WebUI 场景中，用户更习惯"点选上传"而非"打字输入路径"。这个工具更适合有命令行经验的技术用户。

### 5.3 没有 Memory / 上下文管理

Agent 没有"记忆"的概念。每次对话历史只是简单地拼成 messages 列表发给 LLM，没有摘要、没有滑动窗口。

- 对话超过 4096 tokens → LLM 截断 → 丢失早期信息
- 无法做多轮工具调用：第一轮"查模型"→ 第二轮"用这个模型检测这张图"→ Agent 不记得第一轮的结果

### 5.4 工具调用是同步阻塞的

`tool_run_inference` 执行推理时，整个 WebUI 线程被阻塞。模型推理可能耗时几百毫秒到几秒，用户界面会卡住。

改进方向：异步执行 + 进度提示。

### 5.5 没有工具执行的中间反馈

用户输入"帮我检测图片"后，界面没有任何"正在执行推理..."的反馈。如果模型加载慢，用户会以为系统没反应。

### 5.6 路由表扩展性

当前路由表是硬编码的 `_route()` 调用。如果要加一个新工具，需要：
1. 写工具函数
2. 加一行 `_route()` 注册
3. 如果在推理分支有特殊逻辑，还要改 `run_agent()` 里的 if 判断

没有统一注册表——推理分支和通用路由走的不是同一套逻辑。

---

## 六、与主流 Agent 方案对比

| 维度 | ODPlatform Agent | OpenAI Function Calling | LangChain Agent |
|------|------------------|----------------------|----------------|
| **工具选择** | 正则匹配 | LLM 自主选择 | LLM 自主选择 |
| **准确性** | 确定（100% 匹配） | 依赖模型能力 | 依赖模型能力 |
| **灵活性** | 低（仅匹配关键词） | 高（自然语言理解） | 高 |
| **上下文长度消耗** | 0（不传 tools schema） | 高（每次传 tools 定义） | 高 |
| **可调试性** | 高（正则匹配可预测） | 低（LLM 决策不可预测） | 低 |
| **冷启动** | 不需要 LLM | 必须 LLM | 必须 LLM |
| **实现复杂度** | 336 行，纯 Python | 依赖 SDK | 依赖框架 |

---

## 七、总结

ODPlatform 的 Agent 是一个**轻量级、确定性的工具调用系统**，关键词驱动，不依赖 LLM 自身的 function calling 能力。

它的设计哲学是：**让 LLM 做它擅长的事（自然语言润色），不让 LLM 做它不擅长的事（决策调哪个工具）。**

这个设计最适合：
- 工具数量少（< 10 个）
- 用户意图可枚举（查询模型/数据集/实验）
- 需要确定性行为（演示/答辩场景）
- API 调用成本敏感（减少 token 消耗）

如果未来工具数量增长到几十个，或者需要多轮复杂交互，这个架构就需要升级——但 336 行代码的轻量方案，对当前项目规模来说是务实的取舍。

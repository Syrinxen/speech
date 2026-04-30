# SpeechEmotionRecognition-Pytorch

> 作者说明：这个仓库原本是一个偏研究型的语音情绪识别项目，我在当前版本中主动做了收敛和重构，把目标从“训练框架”调整为“可落地的业务能力底座”。  
> 现在它服务于一个更明确的应用场景：**情绪文本还原** 与 **情绪强度评估**。

---

## 1. 项目定位

这个项目面向“情绪回声”一类心理疏导对话系统，负责提供底层情绪理解能力。

如果从产品视角来描述，这个仓库主要承担四件事：

1. 接收音频输入并完成语音转写
2. 对音频做情绪识别
3. 将情绪置信度映射为可解释的强度等级
4. 结合识别结果对文本进行“情绪文本还原”

我没有把仓库继续做成一个大而全的训练平台，而是有意识地把它收缩成一个**更容易部署、更容易接口化、更适合系统集成**的服务底座。原因很简单：

- 在业务落地阶段，接口稳定性通常比训练灵活性更重要
- 对接上层系统时，调用方更关心“输入什么、返回什么、能不能直接用”
- 项目如果同时承担训练、评估、数据处理、在线服务四套职责，维护成本会非常高

因此，这个版本的核心目标不是“覆盖更多算法实验”，而是：

> 用尽量清晰的工程结构，把语音情绪识别能力包装成一套可调用、可解释、可扩展的接口能力。

---

## 2. 当前版本解决什么问题

围绕心理疏导、多模态情绪交互、陪伴式对话等场景，这个项目当前重点解决两类业务问题。

### 2.1 情绪文本还原

现实场景里，用户说出来的话往往并不是结构化、标准化的文本，尤其在情绪波动时更明显。  
比如语句会出现：

- 断裂
- 重复
- 缺少标点
- 情绪含义隐藏在语气里而不是字面里

因此我在当前版本里加入了“情绪文本还原”能力，用于把原始文本转成更适合后续分析和对话策略生成的表达结果。

它并不是文学改写，而是偏工程语义层面的整理，核心产出包括：

- 规范化文本
- 带情绪前缀的还原文本
- 当前表达风格
- 后续沟通建议

### 2.2 情绪强度评估

仅有情绪类别还不够。  
在实际业务里，“愤怒”与“非常愤怒”，“悲伤”与“轻微失落”，在处理策略上差异很大。

所以我增加了情绪强度评估层，把模型置信度映射为：

- 百分制情绪分数
- 强度等级编码
- 中文等级名称
- 情绪极性
- 可解释文本说明

这样做的价值是：

- 方便可视化展示
- 方便后续规则引擎使用
- 方便心理干预策略做分层处理

---

## 3. 能力边界

当前仓库保留并增强的能力如下：

- Whisper 语音转写
- emotion2vec 语音情绪识别
- 情绪分数计算
- 情绪强度评估
- 情绪文本还原
- CLI 调用
- HTTP API 调用
- JSON 结果导出

当前仓库**不再承担**的内容包括：

- 本地训练流程
- 数据集构建工具链
- 传统 BiLSTM / BaseModel 训练代码
- 完整离线评估体系

这是一个明确的设计取舍，而不是能力缺失。  
如果后续业务需要重新扩展训练模块，我建议单独拆分成训练仓库，而不是把在线服务和训练逻辑继续混放。

---

## 4. 项目结构

当前仓库的主要结构如下：

```text
SpeechEmotionRecognition-Pytorch/
├── dataset/                  # 示例数据
├── docs/
│   └── usage.md              # 详细使用文档
├── models/                   # 本地缓存的 emotion2vec 模型
├── mser/
│   ├── __init__.py
│   ├── api.py                # FastAPI 服务接口
│   ├── cli.py                # 命令行入口
│   ├── emotion_service.py    # 情绪文本还原 / 强度评估业务层
│   ├── pipeline.py           # 语音转写 + 情绪识别 + 业务结果整合
│   ├── predict.py            # emotion2vec 推理包装
│   └── utils/
│       └── emotion2vec_predict.py
├── infer.py                  # 纯情绪识别脚本
├── serve_api.py              # API 启动脚本
├── speech_emotion_cli.py     # 命令行分析脚本
├── requirements.txt
├── setup.py
└── README.md
```

我在代码组织上遵循了一条很简单的原则：

- 模型推理和业务规则要分开
- 命令行入口和 HTTP 入口都要能复用同一条分析链路
- 业务系统真正关心的结果，必须在 pipeline 层直接产出

因此：

- `mser/predict.py` 负责“模型级输出”
- `mser/emotion_service.py` 负责“业务级解释”
- `mser/pipeline.py` 负责“端到端结果组合”
- `mser/api.py` 负责“服务化暴露”

---

## 5. 技术路线

### 5.1 语音转写

使用 `openai-whisper` 完成语音转写。  
它负责将输入音频还原为文本，并返回识别语言。

### 5.2 情绪识别

使用 `emotion2vec` 做 utterance 级别的语音情绪识别。  
当前默认模型为：

- `iic/emotion2vec_plus_base`

同时支持：

- `iic/emotion2vec_plus_seed`
- `iic/emotion2vec_plus_large`

### 5.3 情绪强度评估

当前版本将模型的 Top1 置信度转换为百分制 `emotion_score`，并映射到以下等级：

- `very_high` / 极强
- `high` / 高
- `medium` / 中
- `low` / 低
- `very_low` / 很低

这层逻辑目前是规则映射，优点是：

- 可解释
- 轻量
- 易于后续接入业务阈值

### 5.4 情绪文本还原

这层能力不是做“大模型润色”，而是做“面向业务的表达修复”。  
当前实现会基于：

- 文本规范化
- 语言识别
- 情绪标签
- 情绪强度

来生成：

- `normalized_text`
- `restored_text`
- `speaking_style`
- `suggestions`

如果接口只传文本、不传情绪标签，系统会先基于关键词做一个轻量情绪推断，再补齐文本还原和强度评估结果。

---

## 6. 安装说明

建议使用 Python 3.8+。

### 6.1 安装依赖

```bash
pip install -r requirements.txt
```

### 6.2 安装为命令行工具

```bash
pip install .
```

### 6.3 环境要求

请提前确认以下依赖：

- 已安装与当前机器匹配的 PyTorch
- 已安装 FFmpeg
- CPU 可运行，GPU 可选

Whisper 依赖 `ffmpeg`，如果系统命令行无法直接执行 `ffmpeg`，语音转写会失败。

---

## 7. 快速开始

## 7.1 命令行完整分析

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --use_gpu false
```

输出会包含：

- 转写文本
- 识别语言
- 情绪标签
- 置信度
- 情绪分数
- 强度等级
- 文本还原结果

### 7.2 导出 JSON

```bash
python speech_emotion_cli.py \
  --audio_path dataset/test.wav \
  --output_path output/result.json \
  --use_gpu false
```

### 7.3 仅做情绪识别

```bash
python infer.py --audio_path dataset/test.wav --use_gpu false
```

### 7.4 启动 HTTP 服务

```bash
python serve_api.py
```

默认监听：

- `http://127.0.0.1:8000`

接口文档：

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 健康检查: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 8. API 设计

当前服务提供以下接口。

### 8.1 健康检查

`GET /health`

用于确认服务是否正常启动。

返回示例：

```json
{
  "status": "ok",
  "service": "emotion-echo-api"
}
```

### 8.2 本地音频路径分析

`POST /api/v1/emotion/analyze/audio-path`

请求示例：

```json
{
  "audio_path": "dataset/test.wav",
  "emotion_model": "iic/emotion2vec_plus_base",
  "whisper_model": "base",
  "language": "zh",
  "use_gpu": false
}
```

适用场景：

- 服务部署机可以直接访问音频文件
- 本地调试
- 数据处理脚本批量调用

### 8.3 上传音频分析

`POST /api/v1/emotion/analyze/upload`

请求类型：

- `multipart/form-data`

必填字段：

- `file`

可选字段：

- `emotion_model`
- `whisper_model`
- `language`
- `use_gpu`

适用场景：

- 前端上传音频
- 第三方服务直接传文件
- 无法共享本地路径的跨服务调用

### 8.4 情绪文本还原

`POST /api/v1/emotion/restore-text`

请求示例：

```json
{
  "text": "我今天真的很难受 我不知道怎么办",
  "emotion": "sad",
  "confidence": 0.88,
  "language": "zh"
}
```

如果不传 `emotion` 或 `confidence`，系统会基于关键词做轻量推断。

### 8.5 情绪强度评估

`POST /api/v1/emotion/evaluate-intensity`

请求示例：

```json
{
  "emotion": "angry",
  "confidence": 0.91
}
```

返回示例：

```json
{
  "score": 91.0,
  "confidence": 0.91,
  "level_code": "very_high",
  "level_name": "极强",
  "description": "情绪表达非常强烈，建议优先关注触发事件和安抚策略。",
  "valence": "negative",
  "primary_emotion": "angry",
  "primary_emotion_name": "愤怒"
}
```

---

## 9. 输出字段说明

完整音频分析接口会返回以下几类字段。

### 9.1 基础识别字段

- `audio_path`：音频绝对路径
- `transcript`：语音转写文本
- `detected_language`：识别语种
- `emotion`：Top1 情绪标签
- `confidence`：Top1 情绪置信度
- `emotion_score`：百分制情绪得分

### 9.2 排序字段

- `emotion_ranking`：候选情绪列表，包含 `label`、`confidence`、`score`

### 9.3 强度评估字段

- `intensity.score`
- `intensity.level_code`
- `intensity.level_name`
- `intensity.description`
- `intensity.valence`
- `intensity.primary_emotion_name`

### 9.4 文本还原字段

- `text_restoration.normalized_text`
- `text_restoration.restored_text`
- `text_restoration.speaking_style`
- `text_restoration.suggestions`

---

## 10. 适用场景建议

从作者视角来说，我认为这个项目比较适合以下场景：

- 心理疏导对话系统
- 陪伴式聊天机器人
- 情绪监测面板
- 教育陪练中的情绪反馈
- 呼叫中心音频情绪质检

如果你的系统已经有 ASR，那么你也可以只使用：

- `/api/v1/emotion/restore-text`
- `/api/v1/emotion/evaluate-intensity`

把这里当成一个纯业务情绪解释服务。

---

## 11. 当前实现的优点与局限

### 优点

- 结构清晰，便于二次开发
- 已服务化，适合集成
- 输出可解释，适合产品展示
- 兼顾音频入口和文本入口

### 局限

- 情绪文本还原目前主要基于规则增强，不是大模型生成式重写
- 文本情绪推断是轻量关键词策略，适合兜底，不适合替代完整文本情感模型
- 当前强度评估使用置信度映射，适合工程落地，但不等同于临床级心理量表

所以我建议把这个项目理解为：

> 一个可靠的工程底座，而不是一个已经覆盖全部研究问题的最终系统。

---

## 12. 后续扩展建议

如果继续往前做，我建议优先扩展这几个方向：

1. 引入更细粒度的文本情感模型，替代关键词兜底策略
2. 增加情绪趋势跟踪，支持多轮对话中的连续变化分析
3. 引入事件抽取，补齐“什么事导致了什么情绪”
4. 增加数据库存储层，支持用户画像与情绪历史检索
5. 增加前端可视化面板，直接展示强度曲线和情绪摘要

---

## 13. 使用建议

如果你是课程设计、毕设、原型系统开发或企业 PoC，我建议：

- 先直接起 API 服务
- 先打通音频分析接口
- 再根据业务场景调整 `emotion_service.py` 里的规则层

如果你是研究导向用户，我建议把这个仓库当成：

- 推理层
- 业务层
- 服务层

的基础框架，再外接自己的训练仓库或更复杂的文本理解模块。

---

## 14. 详细文档

更完整的使用说明、参数说明、调用示例和排障建议，请查看：

- [docs/usage.md](/D:/SpeechEmotionRecognition-Pytorch/docs/usage.md)

---

## 15. 结语

从作者角度来说，这次调整的重点不是“把项目做大”，而是“把项目做实”。  
我希望这个版本能更接近真实系统中的一个模块，而不仅仅是一个模型演示仓库。

如果你要继续扩展它，我建议优先保持这三件事：

- 输入输出稳定
- 业务解释清晰
- 模型层和业务层边界明确

这会让项目在后续演进时轻松很多。

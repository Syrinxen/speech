# 使用文档

> 作者说明：这份文档不是简单参数罗列，而是按“真正要把项目跑起来并接入系统”的思路来写。  
> 如果 README 负责讲清楚“这是什么”，那么这份文档负责讲清楚“怎么用、怎么接、怎么改”。

---

## 1. 文档目标

本篇文档重点回答以下问题：

1. 项目运行前需要准备什么
2. 命令行如何调用
3. HTTP 接口如何调用
4. 返回结果应该怎样理解
5. 常见问题怎么排查
6. 如果要做业务改造，应该从哪里下手

---

## 2. 环境准备

## 2.1 Python 版本

建议：

- Python 3.8 及以上

## 2.2 必要依赖

安装依赖：

```bash
pip install -r requirements.txt
```

当前依赖主要包括：

- `funasr`
- `modelscope`
- `openai-whisper`
- `fastapi`
- `uvicorn`
- `python-multipart`
- `loguru`

## 2.3 PyTorch

本仓库不强行绑定某一个 PyTorch 安装方式，请根据你的设备环境自行安装适配版本。

例如：

- CPU 环境安装 CPU 版 PyTorch
- NVIDIA GPU 环境安装匹配 CUDA 的 PyTorch

## 2.4 FFmpeg

Whisper 依赖 FFmpeg。  
请确保以下命令可以在终端中正常执行：

```bash
ffmpeg -version
```

如果这个命令不可用，语音转写部分通常会失败。

---

## 3. 模型说明

当前项目默认使用：

- `iic/emotion2vec_plus_base`

也支持：

- `iic/emotion2vec_plus_seed`
- `iic/emotion2vec_plus_large`

模型第一次使用时，如果本地不存在，会自动下载到 `models/` 目录下。

### 3.1 关于 Whisper 模型

Whisper 模型通过参数 `whisper_model` 指定，常见可选值包括：

- `tiny`
- `base`
- `small`
- `medium`
- `large`

建议：

- 本地快速调试时使用 `tiny` 或 `base`
- 追求转写质量时使用更大模型

---

## 4. 快速运行

## 4.1 运行命令行分析

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --use_gpu false
```

它会执行完整流程：

1. 读取音频
2. Whisper 转写
3. emotion2vec 情绪识别
4. 强度评估
5. 文本还原
6. 输出终端结果

### 4.2 示例输出

```text
audio_path: D:\SpeechEmotionRecognition-Pytorch\dataset\test.wav
transcript: Kids are talking by the door.
detected_language: en
emotion: neutral
confidence: 0.91
emotion_score: 91.00
intensity_level: 极强 (very_high)
emotion_description: 情绪表达非常强烈，建议优先关注触发事件和安抚策略。
restored_text: I want to explain this calmly, Kids are talking by the door.
top3_emotions: neutral:91.00, happy:5.00, sad:4.00
```

注意：

- `emotion_score` 来自置信度映射
- `restored_text` 是业务文本还原结果，不是原始 ASR 文本

### 4.3 导出 JSON

```bash
python speech_emotion_cli.py \
  --audio_path dataset/test.wav \
  --output_path output/result.json \
  --use_gpu false
```

适合：

- 批量实验
- 结果留档
- 与其他系统做文件级集成

### 4.4 纯情绪识别

```bash
python infer.py --audio_path dataset/test.wav --use_gpu false
```

这个入口只保留最小输出：

- Top1 情绪标签
- Top1 置信度

适合你只想快速验证 emotion2vec 是否正常工作。

---

## 5. 参数说明

## 5.1 `speech_emotion_cli.py`

完整命令：

```bash
python speech_emotion_cli.py \
  --audio_path dataset/test.wav \
  --emotion_model iic/emotion2vec_plus_base \
  --whisper_model base \
  --language zh \
  --use_gpu false \
  --output_path output/result.json
```

参数说明：

- `--audio_path`
  - 待分析音频路径
  - 必填

- `--emotion_model`
  - emotion2vec 模型名称
  - 默认值：`iic/emotion2vec_plus_base`

- `--whisper_model`
  - Whisper 模型大小
  - 默认值：`base`

- `--language`
  - 强制指定 ASR 语言
  - 可选，例如 `zh`、`en`

- `--use_gpu`
  - 是否启用 GPU
  - 可传 `true/false`

- `--output_path`
  - 可选 JSON 输出路径

## 5.2 `infer.py`

完整命令：

```bash
python infer.py \
  --audio_path dataset/test.wav \
  --emotion_model iic/emotion2vec_plus_base \
  --use_gpu false
```

这个入口不做转写，也不做业务文本还原，仅做情绪识别。

---

## 6. HTTP 服务使用

## 6.1 启动服务

```bash
python serve_api.py
```

或者安装后直接：

```bash
mser-api
```

默认地址：

- `http://127.0.0.1:8000`

接口文档：

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 7. API 详解

## 7.1 健康检查

接口：

```http
GET /health
```

调用示例：

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{
  "status": "ok",
  "service": "emotion-echo-api"
}
```

---

## 7.2 本地音频路径分析

接口：

```http
POST /api/v1/emotion/analyze/audio-path
Content-Type: application/json
```

请求示例：

```json
{
  "audio_path": "dataset/test.wav",
  "emotion_model": "iic/emotion2vec_plus_base",
  "whisper_model": "tiny",
  "language": null,
  "use_gpu": false
}
```

`curl` 示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/emotion/analyze/audio-path" ^
  -H "Content-Type: application/json" ^
  -d "{\"audio_path\":\"dataset/test.wav\",\"whisper_model\":\"tiny\",\"use_gpu\":false}"
```

返回内容包括：

- ASR 文本
- 识别语言
- Top1 情绪
- 全量候选排序
- 强度结果
- 文本还原结果

适合：

- 本地开发
- 服务端已有共享存储
- 后台批处理任务

---

## 7.3 上传音频分析

接口：

```http
POST /api/v1/emotion/analyze/upload
Content-Type: multipart/form-data
```

表单字段：

- `file`：必填，上传音频文件
- `emotion_model`：可选
- `whisper_model`：可选
- `language`：可选
- `use_gpu`：可选

`curl` 示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/emotion/analyze/upload" ^
  -F "file=@dataset/test.wav" ^
  -F "whisper_model=tiny" ^
  -F "use_gpu=false"
```

适合：

- 前端页面上传
- 小程序 / App 上传音频
- 不方便共享音频文件路径的调用场景

---

## 7.4 情绪文本还原

接口：

```http
POST /api/v1/emotion/restore-text
Content-Type: application/json
```

请求示例：

```json
{
  "text": "我今天真的很难受 我不知道怎么办",
  "emotion": "sad",
  "confidence": 0.88,
  "language": "zh"
}
```

返回示例：

```json
{
  "emotion": "sad",
  "emotion_name": "悲伤",
  "confidence": 0.88,
  "emotion_score": 88.0,
  "intensity": {
    "score": 88.0,
    "confidence": 0.88,
    "level_code": "high",
    "level_name": "高",
    "description": "情绪比较强烈，已经明显影响表达方式和沟通节奏。",
    "valence": "negative",
    "primary_emotion": "sad",
    "primary_emotion_name": "悲伤"
  },
  "text_restoration": {
    "normalized_text": "我今天真的很难受我不知道怎么办。",
    "restored_text": "我现在心里挺难受的，我今天真的很难受我不知道怎么办。",
    "speaking_style": "悲伤、高强度",
    "suggestions": [
      "建议补充最近一次触发低落情绪的时间点和场景。",
      "如果用于对话干预，可优先给出共情式回应，避免直接讲道理。"
    ]
  },
  "inferred_from_text": false
}
```

### 说明

如果只传：

```json
{
  "text": "我真的很难受"
}
```

系统会先做轻量情绪推断，再补全强度和还原结果。  
这适合作为兜底能力，但我仍然建议在完整业务系统里优先传入更可信的情绪标签和置信度。

---

## 7.5 情绪强度评估

接口：

```http
POST /api/v1/emotion/evaluate-intensity
Content-Type: application/json
```

请求示例：

```json
{
  "emotion": "angry",
  "confidence": 0.91
}
```

`curl` 示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/emotion/evaluate-intensity" ^
  -H "Content-Type: application/json" ^
  -d "{\"emotion\":\"angry\",\"confidence\":0.91}"
```

这个接口适合作为独立的规则层服务，被外部模型结果直接调用。

---

## 8. 返回结构说明

完整音频分析接口一般返回以下结构：

```json
{
  "audio_path": "D:/SpeechEmotionRecognition-Pytorch/dataset/test.wav",
  "transcript": "Kids are talking by the door.",
  "detected_language": "en",
  "emotion": "neutral",
  "confidence": 0.91,
  "emotion_score": 91.0,
  "emotion_ranking": [
    {
      "label": "neutral",
      "confidence": 0.91,
      "score": 91.0
    }
  ],
  "intensity": {
    "score": 91.0,
    "confidence": 0.91,
    "level_code": "very_high",
    "level_name": "极强",
    "description": "情绪表达非常强烈，建议优先关注触发事件和安抚策略。",
    "valence": "neutral",
    "primary_emotion": "neutral",
    "primary_emotion_name": "平静"
  },
  "text_restoration": {
    "normalized_text": "Kids are talking by the door.",
    "restored_text": "I want to explain this calmly, Kids are talking by the door.",
    "speaking_style": "neutral with very_high intensity",
    "suggestions": [
      "Collect more context about the event, people involved, and timeline.",
      "Ask for one recent example to make the emotional signal easier to interpret."
    ]
  }
}
```

---

## 9. 业务改造建议

如果你要把这个项目改成自己的业务版本，我建议按下面顺序修改。

## 9.1 调整文本还原规则

文件：

- `mser/emotion_service.py`

这里可以改：

- 情绪中文名
- 强度文案
- 前缀表达
- 沟通建议
- 文本情绪关键词

如果你的应用场景不是心理疏导，而是客服、教育、陪伴机器人，那么这里是最值得优先定制的地方。

## 9.2 替换文本情绪推断策略

当前的文本情绪推断是关键词兜底逻辑，优点是轻量，缺点是泛化有限。  
如果你希望文本入口更强，可以考虑：

- 接入情感分类模型
- 接入大语言模型做结构化情绪抽取
- 引入多标签情绪识别

## 9.3 增加数据库层

当前项目默认无数据库。  
如果要做真实系统，建议增加：

- 用户表
- 情绪记录表
- 对话记录表
- 反馈记录表

## 9.4 增加鉴权

如果要提供外部服务，建议为 API 增加：

- Token 鉴权
- 调用日志
- 限流
- 异常审计

---

## 10. Python 调用示例

## 10.1 调用文本还原接口

```python
import requests

payload = {
    "text": "我今天真的很难受 我不知道怎么办",
    "emotion": "sad",
    "confidence": 0.88,
    "language": "zh",
}

resp = requests.post(
    "http://127.0.0.1:8000/api/v1/emotion/restore-text",
    json=payload,
    timeout=60,
)

print(resp.json())
```

## 10.2 调用音频分析接口

```python
import requests

payload = {
    "audio_path": "dataset/test.wav",
    "whisper_model": "tiny",
    "use_gpu": False,
}

resp = requests.post(
    "http://127.0.0.1:8000/api/v1/emotion/analyze/audio-path",
    json=payload,
    timeout=300,
)

print(resp.json())
```

---

## 11. 错误处理说明

### 11.1 音频文件不存在

返回：

- HTTP `404`

示例：

```json
{
  "detail": "Audio file does not exist: missing.wav"
}
```

### 11.2 依赖缺失

常见原因：

- 未安装 `fastapi`
- 未安装 `uvicorn`
- 未安装 `openai-whisper`
- 未安装 FFmpeg
- 未安装匹配版本的 PyTorch

建议先执行：

```bash
pip install -r requirements.txt
```

### 11.3 首次运行较慢

首次调用时可能出现较慢情况，常见原因是：

- emotion2vec 模型首次下载
- Whisper 模型首次下载
- CPU 推理速度本身较慢

这属于正常现象。

---

## 12. 性能与部署建议

### 12.1 开发调试

建议：

- `whisper_model=tiny`
- `use_gpu=false`

优点：

- 启动快
- 占用低
- 更适合本地调试接口

### 12.2 演示环境

建议：

- `whisper_model=base`
- 优先使用 GPU

优点：

- 转写质量更稳
- 演示体验更好

### 12.3 生产化改造

建议：

- 增加鉴权
- 增加日志
- 增加进程管理
- 增加模型预热
- 增加任务队列

如果后续并发量上来，仅依赖单进程 FastAPI 入口通常不够，需要再加：

- Gunicorn / 多 worker 管理
- 任务异步化
- 推理资源隔离

---

## 13. 常见问题

### 13.1 为什么识别到的情绪和文本直觉不完全一致

因为当前主情绪来自语音模型，不是纯文本模型。  
语音中的语速、音高、力度、停顿会影响最终判断。

### 13.2 为什么“平静”也可能出现高强度

因为当前“强度”严格来说更接近“情绪表达显著程度”与“模型置信程度”的结合映射，而不是只表示负面情绪的剧烈程度。  
所以平静也可能有高置信度，从而映射到更高强度级别。

### 13.3 情绪文本还原是不是大模型生成

不是。  
当前版本是规则增强式业务还原，追求可控、可解释、易部署。

### 13.4 可不可以只用文本接口，不用音频接口

可以。  
如果你已经有自己的 ASR，可以直接调用：

- `/api/v1/emotion/restore-text`
- `/api/v1/emotion/evaluate-intensity`

---

## 14. 建议阅读顺序

如果你第一次接触这个项目，我建议这样看：

1. 先看 `README.md`，了解项目定位
2. 跑一次 `speech_emotion_cli.py`
3. 启动 `serve_api.py`
4. 打开 `/docs` 调一次接口
5. 最后再去改 `mser/emotion_service.py`

这样理解成本最低。

---

## 15. 总结

这个项目现在最重要的价值，不是“模型很多”，而是：

- 结构够清楚
- 输出够直接
- 接口够可用

如果你的目标是做一个课程项目、系统原型、业务 PoC，当前版本已经可以直接作为底层服务使用。  
如果你的目标是继续研究更强的情绪理解能力，那么也完全可以把这里当成一个稳定的工程起点。

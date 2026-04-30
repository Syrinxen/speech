# 使用说明

## 1. 运行前准备

### 1.1 Python

- 建议 `Python 3.8+`

### 1.2 安装依赖

```bash
pip install -r requirements.txt
```

如果你希望把命令安装到环境里：

```bash
pip install .
```

安装后可直接使用：

- `mser-speech`
- `mser-api`

### 1.3 PyTorch

仓库不绑定具体的 PyTorch 发行方式，请按你的硬件环境自行安装：

- CPU 环境安装 CPU 版
- NVIDIA GPU 环境安装匹配 CUDA 的版本

### 1.4 FFmpeg

Whisper 依赖 `ffmpeg`，请先确认以下命令可用：

```bash
ffmpeg -version
```

如果这一步失败，语音转写通常也会失败。

## 2. 模型与下载行为

默认情绪模型：

- `iic/emotion2vec_plus_base`

也支持：

- `iic/emotion2vec_plus_seed`
- `iic/emotion2vec_plus_large`

Whisper 模型通过 `whisper_model` 指定，常见值包括：

- `tiny`
- `base`
- `small`
- `medium`
- `large`

首次运行时，如果本地还没有模型，相关依赖会自动下载缓存。开发调试建议先用 `tiny` 或 `base`，便于缩短冷启动时间。

## 3. 最常用的三种运行方式

### 3.1 完整音频分析

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --use_gpu false
```

这个入口会依次执行：

1. 读取音频
2. Whisper 转写
3. `emotion2vec` 情绪识别
4. 情绪强度评估
5. 文本修复
6. 打印结构化结果

### 3.2 导出结果到 JSON

```bash
python speech_emotion_cli.py ^
  --audio_path dataset/test.wav ^
  --output_path output/result.json ^
  --use_gpu false
```

适合：

- 留存分析结果
- 跑批量实验
- 对接下游流程

### 3.3 只验证情绪识别

```bash
python infer.py --audio_path dataset/test.wav --use_gpu false
```

这个入口只保留最小输出：

- `emotion`
- `confidence`

如果你只想验证模型能不能正常识别音频，用它最快。

## 4. CLI 参数说明

### 4.1 `speech_emotion_cli.py`

完整示例：

```bash
python speech_emotion_cli.py ^
  --audio_path dataset/test.wav ^
  --emotion_model iic/emotion2vec_plus_base ^
  --whisper_model base ^
  --language zh ^
  --use_gpu false ^
  --output_path output/result.json
```

主要参数：

- `--audio_path`：待分析音频路径，必填
- `--emotion_model`：情绪模型名称
- `--whisper_model`：Whisper 模型规格
- `--language`：可选，强制指定转写语言，例如 `zh` 或 `en`
- `--use_gpu`：是否启用 GPU，支持 `true/false`
- `--output_path`：可选，输出 JSON 文件路径

### 4.2 `infer.py`

完整示例：

```bash
python infer.py ^
  --audio_path dataset/test.wav ^
  --emotion_model iic/emotion2vec_plus_base ^
  --use_gpu false
```

它不会做 Whisper 转写，也不会做业务层文本修复。

## 5. HTTP API 使用

### 5.1 启动服务

```bash
python serve_api.py
```

或者安装后直接启动：

```bash
mser-api
```

默认地址：

- [http://127.0.0.1:8000](http://127.0.0.1:8000)
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5.2 健康检查

```bash
curl http://127.0.0.1:8000/health
```

返回：

```json
{
  "status": "ok",
  "service": "emotion-echo-api"
}
```

### 5.3 分析本地音频路径

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

适合：

- 本地开发联调
- 服务端能直接访问共享文件
- 后台批处理任务

### 5.4 上传音频文件

接口：

```http
POST /api/v1/emotion/analyze/upload
Content-Type: multipart/form-data
```

`curl` 示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/emotion/analyze/upload" ^
  -F "file=@dataset/test.wav" ^
  -F "whisper_model=tiny" ^
  -F "use_gpu=false"
```

适合：

- 前端直接上传文件
- 小程序或 App 上传语音
- 无法共享本地路径的跨服务调用

### 5.5 文本修复接口

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

如果你不传 `emotion` 或 `confidence`，服务会先基于关键词做一次轻量推断，再补齐强度评估和修复结果。

### 5.6 情绪强度评估接口

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

这个接口保留的原因是它属于业务解释层能力，不是测试集评测逻辑。

## 6. 返回结果怎么理解

完整音频分析返回值里最常看的字段有：

- `audio_path`：解析后的绝对路径
- `transcript`：转写文本
- `detected_language`：识别语言
- `emotion`：Top1 情绪标签
- `confidence`：Top1 置信度
- `emotion_score`：百分制分数
- `emotion_ranking`：候选情绪排序列表
- `intensity.level_code`：强度编码
- `intensity.level_name`：强度中文名
- `intensity.description`：强度解释文案
- `text_restoration.normalized_text`：规范化文本
- `text_restoration.restored_text`：修复后的表达文本
- `text_restoration.suggestions`：后续沟通建议

一个典型返回结构如下：

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

## 7. 常见问题

### 7.1 音频文件不存在

接口会返回 `404`，示例：

```json
{
  "detail": "Audio file does not exist: missing.wav"
}
```

### 7.2 第一次运行很慢

通常是以下原因：

- `emotion2vec` 首次下载
- Whisper 首次下载
- CPU 推理本来就比 GPU 慢

这属于正常现象。

### 7.3 为什么识别情绪和文本直觉不完全一致

因为主情绪判断来自语音信号，不是纯文本情感分类。语速、停顿、音高和能量变化都会影响最终结果。

### 7.4 为什么“平静”也可能出现高强度

当前“强度”更接近“表达显著程度 + 模型置信度”的工程映射，不等同于负面情绪烈度。

## 8. 建议从哪里改

如果你准备继续做业务化开发，推荐按下面顺序看代码：

1. [README.md](/D:/SpeechEmotionRecognition-Pytorch/README.md)
2. [mser/cli.py](/D:/SpeechEmotionRecognition-Pytorch/mser/cli.py)
3. [mser/pipeline.py](/D:/SpeechEmotionRecognition-Pytorch/mser/pipeline.py)
4. [mser/emotion_service.py](/D:/SpeechEmotionRecognition-Pytorch/mser/emotion_service.py)
5. [mser/api.py](/D:/SpeechEmotionRecognition-Pytorch/mser/api.py)

优先改造点：

- 业务文案和建议话术：`mser/emotion_service.py`
- 输出结构和串联流程：`mser/pipeline.py`
- 服务化能力和接口约束：`mser/api.py`
- 模型替换与扩展：`mser/predict.py`

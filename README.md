# SpeechEmotionRecognition-Pytorch

基于 `emotion2vec` 和 Whisper 的语音情绪识别项目，提供命令行入口、可复用的 Python 流水线以及 FastAPI 服务接口。

这个仓库当前聚焦两件事：

- 把一段音频转成可消费的结构化结果，包括转写文本、情绪标签、置信度和排序结果
- 在识别结果之上补一层业务解释，包括情绪强度评估和文本表达修复

## 项目能力

- Whisper 语音转写
- `emotion2vec` 语音情绪识别
- 情绪置信度到强度等级的规则映射
- 面向对话场景的文本修复结果生成
- CLI 调用
- HTTP API 调用
- JSON 结果导出

## 仓库结构

```text
SpeechEmotionRecognition-Pytorch/
|-- dataset/
|   `-- test.wav
|-- docs/
|   `-- usage.md
|-- mser/
|   |-- api.py
|   |-- cli.py
|   |-- emotion_service.py
|   |-- pipeline.py
|   |-- predict.py
|   `-- utils/
|-- infer.py
|-- requirements.txt
|-- serve_api.py
|-- setup.py
`-- speech_emotion_cli.py
```

核心模块说明：

- `mser/pipeline.py`：端到端音频分析流水线
- `mser/predict.py`：`emotion2vec` 推理封装
- `mser/emotion_service.py`：强度评估和文本修复规则
- `mser/api.py`：FastAPI 服务入口
- `mser/cli.py`：命令行入口

## 环境要求

- Python `3.8+`
- 本机可用的 PyTorch
- `ffmpeg`

安装依赖：

```bash
pip install -r requirements.txt
```

如果要安装为命令行工具：

```bash
pip install .
```

## 快速开始

完整音频分析：

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --use_gpu false
```

只做情绪识别：

```bash
python infer.py --audio_path dataset/test.wav --use_gpu false
```

导出 JSON：

```bash
python speech_emotion_cli.py ^
  --audio_path dataset/test.wav ^
  --output_path output/result.json ^
  --use_gpu false
```

启动 API：

```bash
python serve_api.py
```

服务启动后默认可访问：

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## API 概览

### `GET /health`

服务健康检查。

### `POST /api/v1/emotion/analyze/audio-path`

输入本地音频路径，返回：

- 转写文本
- 识别语言
- Top1 情绪
- 全量候选排序
- 情绪强度结果
- 文本修复结果

### `POST /api/v1/emotion/analyze/upload`

上传音频文件做同样的完整分析，适合前端上传和跨服务调用。

### `POST /api/v1/emotion/restore-text`

对已有文本做情绪化表达修复；如果调用方没有传情绪和置信度，会先做一次轻量推断。

### `POST /api/v1/emotion/evaluate-intensity`

根据情绪标签和置信度返回强度等级解释。这是业务层能力，不是离线测试集评测逻辑。

## 输出结果说明

完整分析结果包含以下关键字段：

- `transcript`：Whisper 转写文本
- `detected_language`：识别语言
- `emotion`：Top1 情绪标签
- `confidence`：Top1 置信度
- `emotion_score`：百分制分数
- `emotion_ranking`：候选情绪排序
- `intensity`：强度等级、情绪极性和解释文案
- `text_restoration`：规范化文本、修复文本、表达风格和建议

## 二次开发建议

如果你要在这个仓库上继续做业务化改造，优先看这几个位置：

- [mser/emotion_service.py](/D:/SpeechEmotionRecognition-Pytorch/mser/emotion_service.py)：修改强度阈值、中文文案、修复前缀、建议话术
- [mser/pipeline.py](/D:/SpeechEmotionRecognition-Pytorch/mser/pipeline.py)：调整完整流水线的输入输出结构
- [mser/api.py](/D:/SpeechEmotionRecognition-Pytorch/mser/api.py)：增加鉴权、日志、请求校验和部署接口
- [mser/predict.py](/D:/SpeechEmotionRecognition-Pytorch/mser/predict.py)：替换或扩展情绪模型推理层

## 文档

更完整的运行说明、参数示例和故障排查见：

- [docs/usage.md](/D:/SpeechEmotionRecognition-Pytorch/docs/usage.md)

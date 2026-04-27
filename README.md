# SpeechEmotionRecognition-Pytorch

基于现有 `mser` 情绪识别能力，这个仓库现在收敛为一条更清晰的主流程：

1. 语音转文字
2. 情绪识别
3. 情绪评分输出
4. JSON 结果落盘

推荐直接使用统一入口 [`speech_emotion_cli.py`](speech_emotion_cli.py) 或安装后的命令 `mser-speech`。

## 核心能力

- 语音转文字：使用 Whisper 对输入音频进行转写
- 情绪识别：复用项目现有 `MSERPredictor`
- 情绪评分：返回 Top1 情绪、置信度和 0-100 分制评分
- 排名结果：返回按概率排序的情绪候选列表
- 结果导出：支持保存为 JSON

## 项目结构

- [`mser/pipeline.py`](mser/pipeline.py)：统一语音分析管线
- [`mser/cli.py`](mser/cli.py)：命令行入口
- [`speech_emotion_cli.py`](speech_emotion_cli.py)：根目录快捷启动脚本
- [`mser/predict.py`](mser/predict.py)：情绪预测器，已补充完整评分接口
- [`docs/usage.md`](docs/usage.md)：详细使用文档

## 安装

建议使用 Python 3.8+。

### 1. 安装 PyTorch

请按你的 CUDA 或 CPU 环境先安装 PyTorch。

### 2. 安装项目依赖

```bash
pip install -r requirements.txt
```

如果你希望作为包方式使用，也可以执行：

```bash
pip install .
```

### 3. 安装 FFmpeg

Whisper 依赖 FFmpeg 读取音频，请确保系统环境中已经可用 `ffmpeg` 命令。

## 快速开始

### 方式一：使用本地情绪模型

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --use_gpu false
```

### 方式二：使用 ModelScope Emotion2Vec 模型

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --use_ms_model iic/emotion2vec_plus_base --use_gpu false
```

### 导出 JSON

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --output_path output/result.json --use_gpu false
```

## 输出示例

```text
audio_path: D:\SpeechEmotionRecognition-Pytorch\dataset\test.wav
transcript: hello everyone
detected_language: en
emotion: Angry
confidence: 0.99995
emotion_score: 100.00
top3_emotions: Angry:100.00, Fear:0.00, Sad:0.00
```

其中：

- `confidence`：模型原始置信度，范围 0-1
- `emotion_score`：将置信度换算为 0-100 分
- `top3_emotions`：前三个候选情绪及分数

## 更多说明

完整使用说明、参数解释和常见问题见 [`docs/usage.md`](docs/usage.md)。

# SpeechEmotionRecognition-Pytorch

项目已精简为只保留 `emotion2vec` 推理链路，不再包含自训练、本地 BiLSTM/BaseModel、特征提取和评估模块。

当前核心功能：

1. 语音转文字
2. 情绪识别
3. 情绪评分
4. JSON 结果导出

推荐入口：

- [`speech_emotion_cli.py`](speech_emotion_cli.py)：完整语音转文字 + 情绪识别
- [`infer.py`](infer.py)：纯情绪识别

## 保留模型

项目默认使用：

- `iic/emotion2vec_plus_base`

同时支持：

- `iic/emotion2vec_plus_seed`
- `iic/emotion2vec_plus_large`

如本地不存在，对应模型会自动下载。

## 安装

建议使用 Python 3.8+，并先按你的环境安装 PyTorch。

安装依赖：

```bash
pip install -r requirements.txt
```

或安装为命令行工具：

```bash
pip install .
```

Whisper 依赖 FFmpeg，请确保系统环境中能直接运行 `ffmpeg`。

## 快速开始

完整流程：

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --use_gpu false
```

指定 emotion2vec 模型：

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --emotion_model iic/emotion2vec_plus_base --use_gpu false
```

导出 JSON：

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --output_path output/result.json --use_gpu false
```

只做情绪识别：

```bash
python infer.py --audio_path dataset/test.wav --use_gpu false
```

## 输出示例

```text
audio_path: D:\SpeechEmotionRecognition-Pytorch\dataset\test.wav
transcript: hello everyone
detected_language: en
emotion: angry
confidence: 0.99995
emotion_score: 100.00
top3_emotions: angry:100.00, fear:0.00, sad:0.00
```

详细说明见 [`docs/usage.md`](docs/usage.md)。

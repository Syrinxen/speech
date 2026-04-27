# 使用文档

## 1. 项目说明

当前仓库只保留 `emotion2vec` 推理能力，已经移除：

- 本地自训练模块
- BiLSTM / BaseModel
- 特征提取流程
- 数据集构建与评估流程

保留下来的核心链路是：

1. Whisper 语音转文字
2. emotion2vec 情绪识别
3. 置信度与百分制评分输出

## 2. 推荐入口

完整流程：

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav
```

安装后也可以使用：

```bash
mser-speech --audio_path dataset/test.wav
```

纯情绪识别：

```bash
python infer.py --audio_path dataset/test.wav
```

## 3. 环境准备

- Python 3.8 及以上
- 预先安装 PyTorch
- 系统已安装 FFmpeg

安装依赖：

```bash
pip install -r requirements.txt
```

## 4. 参数说明

### `speech_emotion_cli.py`

```bash
python speech_emotion_cli.py \
  --audio_path dataset/test.wav \
  --emotion_model iic/emotion2vec_plus_base \
  --whisper_model base \
  --language zh \
  --use_gpu false \
  --output_path output/result.json
```

- `--audio_path`：待分析音频路径
- `--emotion_model`：emotion2vec 模型名称
- `--whisper_model`：Whisper 模型大小
- `--language`：强制指定转写语言，如 `zh`、`en`
- `--use_gpu`：是否启用 GPU
- `--output_path`：将结果保存为 JSON

### `infer.py`

```bash
python infer.py --audio_path dataset/test.wav --emotion_model iic/emotion2vec_plus_base --use_gpu false
```

- `--audio_path`：待分析音频路径
- `--emotion_model`：emotion2vec 模型名称
- `--use_gpu`：是否启用 GPU

## 5. 结果字段

- `transcript`：转写文本
- `detected_language`：识别语言
- `emotion`：Top1 情绪标签
- `confidence`：Top1 置信度，范围 0-1
- `emotion_score`：百分制评分
- `emotion_ranking`：按置信度排序的候选列表

## 6. 故障排查

缺少 `torch`：

先安装与你机器匹配的 PyTorch。

缺少 `whisper` 或 `funasr`：

```bash
pip install -r requirements.txt
```

找不到 `ffmpeg`：

请先安装 FFmpeg，并确保命令行可直接运行 `ffmpeg`。

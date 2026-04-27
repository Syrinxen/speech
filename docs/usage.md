# 使用文档

## 1. 功能概览

本项目面向一个简化后的核心场景：给定一段音频，输出转写文本、情绪标签和情绪评分。

当前推荐流程：

1. 使用 Whisper 做语音转文字
2. 使用 `MSERPredictor` 做情绪识别
3. 输出置信度、百分制评分和情绪排序结果

## 2. 推荐入口

推荐使用以下任一入口：

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav
```

或安装后：

```bash
mser-speech --audio_path dataset/test.wav
```

## 3. 环境准备

### Python

- Python 3.8 及以上

### PyTorch

项目依赖 PyTorch，请先按你的机器环境完成安装。

### 安装依赖

```bash
pip install -r requirements.txt
```

### FFmpeg

Whisper 需要依赖 FFmpeg 处理音频，请确保命令行可以直接运行：

```bash
ffmpeg -version
```

## 4. 命令参数

```bash
python speech_emotion_cli.py \
  --audio_path dataset/test.wav \
  --configs configs/bi_lstm.yml \
  --model_path models/BiLSTM_Emotion2Vec/best_model/ \
  --whisper_model base \
  --language zh \
  --use_gpu false \
  --output_path output/result.json
```

参数说明：

- `--audio_path`：待分析音频路径，必填
- `--configs`：本地情绪模型配置文件路径
- `--model_path`：本地情绪模型目录或 `model.pth` 文件路径
- `--use_ms_model`：使用 ModelScope 的 Emotion2Vec 模型时填写，例如 `iic/emotion2vec_plus_base`
- `--whisper_model`：Whisper 模型大小，可选 `tiny`、`base`、`small`、`medium`、`large`
- `--language`：强制指定转写语言，如 `zh`、`en`；不填则自动检测
- `--use_gpu`：是否启用 GPU，支持 `true/false`
- `--overwrites`：覆盖配置文件中的参数
- `--output_path`：将结果保存为 JSON

## 5. 常见用法

### 5.1 本地情绪模型推理

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --use_gpu false
```

适用场景：

- 仓库中已有训练好的本地模型
- 希望完全基于当前项目落地

### 5.2 使用 ModelScope Emotion2Vec

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --use_ms_model iic/emotion2vec_plus_base --use_gpu false
```

适用场景：

- 想跳过本地训练配置
- 直接使用公开 Emotion2Vec 模型

### 5.3 保存 JSON 结果

```bash
python speech_emotion_cli.py --audio_path dataset/test.wav --output_path output/result.json --use_gpu false
```

示例 JSON：

```json
{
  "audio_path": "D:\\SpeechEmotionRecognition-Pytorch\\dataset\\test.wav",
  "transcript": "hello everyone",
  "detected_language": "en",
  "emotion": "Angry",
  "confidence": 0.99995,
  "emotion_score": 100.0,
  "emotion_ranking": [
    {
      "label": "Angry",
      "confidence": 0.99995,
      "score": 100.0
    }
  ]
}
```

## 6. 输出字段解释

- `transcript`：语音识别文本
- `detected_language`：Whisper 自动检测到的语言
- `emotion`：Top1 情绪标签
- `confidence`：Top1 情绪置信度，范围 0-1
- `emotion_score`：将置信度换算为 0-100 分
- `emotion_ranking`：按置信度从高到低排列的情绪候选

## 7. 与旧脚本的关系

仓库中原有的 `infer.py` 仍可用于纯情绪识别；
新的统一入口更适合完整业务流程，因为它把语音转文字和情绪评分合并到了同一条链路中。

## 8. 故障排查

### 缺少 `torch`

说明 PyTorch 尚未安装，请先安装适合当前环境的 PyTorch。

### 缺少 `whisper`

请重新安装依赖：

```bash
pip install -r requirements.txt
```

### 找不到 `ffmpeg`

请先安装 FFmpeg，并确认 `ffmpeg` 已加入系统环境变量。

### GPU 不可用

将参数改为：

```bash
--use_gpu false
```

## 9. 开发建议

如果后续要继续扩展，建议围绕以下两个层次继续演进：

- `mser/pipeline.py`：统一业务流程
- `mser/predict.py`：模型预测与评分接口

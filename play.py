import whisper
import os
from mser.predict import MSERPredictor

def speech_recognition(audio_path, model_size="base", language=None):
    """
    语音识别（使用 Whisper）
    :param audio_path: 音频文件路径
    :param model_size: 模型大小（tiny/base/small/large）
    :param language: 语言代码（如 'zh' 强制中文，'en' 强制英文，None 自动检测）
    :return: 识别文本
    """
    if not os.path.exists(audio_path):
        return f"错误：文件不存在 → {audio_path}"

    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio=audio_path,
        language=language,      # None 时自动检测
        fp16=False,             # Windows 兼容
        verbose=False,
        initial_prompt=None     # 可根据语言自定义提示
    )
    return result["text"].strip()

def emotion_recognition(audio_path, use_ms_model="iic/emotion2vec_plus_base", use_gpu=True):
    """
    情感识别（使用 Emotion2Vec）
    :param audio_path: 音频文件路径
    :param use_ms_model: Emotion2Vec 模型名称
    :param use_gpu: 是否使用 GPU
    :return: (情感标签, 置信度)
    """
    predictor = MSERPredictor(
        configs=None,               # 使用 use_ms_model 时 configs 可设为 None
        use_ms_model=use_ms_model,
        use_gpu=use_gpu,
        log_level="info"
    )
    label, score = predictor.predict(audio_data=audio_path)
    return label, score

def main(audio_path, whisper_model="base", language=None, emotion_model="iic/emotion2vec_plus_base", use_gpu=True):
    print(f"正在处理音频: {audio_path}")
    
    # 语音识别
    print("正在进行语音识别...")
    text = speech_recognition(audio_path, whisper_model, language)
    print(f"识别文字: {text}")
    
    # 情感识别
    print("正在进行情感识别...")
    emotion, confidence = emotion_recognition(audio_path, emotion_model, use_gpu)
    print(f"情感标签: {emotion}，置信度: {confidence:.4f}")

if __name__ == "__main__":
    # 配置参数（请根据实际情况修改）
    AUDIO_PATH = r"D:\edge下载\a3e7e-main\a3e7e-main\标准语音测试包\标准语音测试包\英语\OSR_us_000_0011_8k.wav"
    WHISPER_MODEL = "base"          # 可选 tiny/base/small/large
    LANGUAGE = None                 # 自动检测语言（英语时自动识别为英文）
    EMOTION_MODEL = "iic/emotion2vec_plus_base"  # 情感模型
    USE_GPU = True                  # 如果有 GPU 设为 True
    
    main(AUDIO_PATH, WHISPER_MODEL, LANGUAGE, EMOTION_MODEL, USE_GPU)
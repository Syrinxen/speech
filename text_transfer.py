import whisper
import os


AUDIO_PATH = r"D:\SpeechEmotionRecognition-Pytorch\dataset\test.wav"
MODEL_SIZE = "base"

def chinese_speech_recognition(audio_path, model_size="base"):
    """
    语音识别（自动适配8k/16k/48k采样率，单/双声道）
    :param audio_path: 音频文件路径（WAV/MP3均可）
    :param model_size: 模型大小（tiny/base/small/large）
    :return: 识别后的字符串
    """

    if not os.path.exists(audio_path):
        return f"错误：文件不存在 → {audio_path}"
    
    model = whisper.load_model(model_size)
    

    result = model.transcribe(
        audio=audio_path,
        fp16=False,            
        verbose=False,         
        initial_prompt="None"  
    )
    
    return result["text"].strip()

if __name__ == "__main__":
    result = chinese_speech_recognition(AUDIO_PATH, MODEL_SIZE)
    print(f"{result}")
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

from mser import DEFAULT_EMOTION2VEC_MODEL
from mser.emotion_service import EmotionBusinessService
from mser.predict import MSERPredictor


@dataclass
class EmotionScore:
    label: str
    confidence: float
    score: float


@dataclass
class EmotionIntensityInfo:
    score: float
    confidence: float
    level_code: str
    level_name: str
    description: str
    valence: str
    primary_emotion: str
    primary_emotion_name: str


@dataclass
class EmotionTextRestoreInfo:
    normalized_text: str
    restored_text: str
    speaking_style: str
    suggestions: List[str]


@dataclass
class SpeechEmotionResult:
    audio_path: str
    transcript: str
    detected_language: Optional[str]
    emotion: str
    confidence: float
    emotion_score: float
    emotion_ranking: List[EmotionScore]
    intensity: EmotionIntensityInfo
    text_restoration: EmotionTextRestoreInfo

    def to_dict(self):
        return asdict(self)


class WhisperSpeechRecognizer:
    def __init__(self, model_size="base", language=None, use_gpu=True):
        self.model_size = model_size
        self.language = language
        self.use_gpu = use_gpu
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            import whisper
        except ImportError as exc:
            raise ImportError(
                "Whisper 转写依赖 `openai-whisper` 和 `torch`，请先安装运行依赖。"
            ) from exc
        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        self._model = whisper.load_model(self.model_size, device=device)
        return self._model

    def transcribe(self, audio_path):
        model = self._load_model()
        result = model.transcribe(
            audio=audio_path,
            language=self.language,
            task="transcribe",
            fp16=False,
            verbose=False,
        )
        return result.get("text", "").strip(), result.get("language")


class SpeechEmotionPipeline:
    def __init__(
        self,
        emotion_model=DEFAULT_EMOTION2VEC_MODEL,
        whisper_model="base",
        language=None,
        use_gpu=True,
    ):
        self.speech_recognizer = WhisperSpeechRecognizer(
            model_size=whisper_model,
            language=language,
            use_gpu=use_gpu,
        )
        self.emotion_recognizer = MSERPredictor(
            emotion_model=emotion_model,
            use_gpu=use_gpu,
        )
        self.business_service = EmotionBusinessService()

    def analyze(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        transcript, detected_language = self.speech_recognizer.transcribe(audio_path)
        emotion_ranking = self.emotion_recognizer.predict_scores(audio_path)
        top_emotion = emotion_ranking[0]
        intensity = self.business_service.evaluate_intensity(
            emotion=top_emotion["label"],
            confidence=top_emotion["confidence"],
        )
        restoration = self.business_service.restore_text(
            text=transcript,
            emotion=top_emotion["label"],
            confidence=top_emotion["confidence"],
            language=detected_language,
        )
        return SpeechEmotionResult(
            audio_path=os.path.abspath(audio_path),
            transcript=transcript,
            detected_language=detected_language,
            emotion=top_emotion["label"],
            confidence=top_emotion["confidence"],
            emotion_score=top_emotion["score"],
            emotion_ranking=[
                EmotionScore(
                    label=item["label"],
                    confidence=item["confidence"],
                    score=item["score"],
                )
                for item in emotion_ranking
            ],
            intensity=EmotionIntensityInfo(**intensity.to_dict()),
            text_restoration=EmotionTextRestoreInfo(**restoration.to_dict()),
        )

    @staticmethod
    def save_result(result, output_path):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

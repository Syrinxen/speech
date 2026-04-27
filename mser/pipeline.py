import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

from mser.predict import MSERPredictor


@dataclass
class EmotionScore:
    label: str
    confidence: float
    score: float


@dataclass
class SpeechEmotionResult:
    audio_path: str
    transcript: str
    detected_language: Optional[str]
    emotion: str
    confidence: float
    emotion_score: float
    emotion_ranking: List[EmotionScore]

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
                "Whisper transcription requires `openai-whisper` and `torch`. "
                "Please install the project dependencies first."
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
        emotion_configs="configs/bi_lstm.yml",
        emotion_model_path="models/BiLSTM_Emotion2Vec/best_model/",
        use_ms_model=None,
        whisper_model="base",
        language=None,
        use_gpu=True,
        overwrites=None,
    ):
        self.speech_recognizer = WhisperSpeechRecognizer(
            model_size=whisper_model,
            language=language,
            use_gpu=use_gpu,
        )
        self.emotion_recognizer = MSERPredictor(
            configs=emotion_configs,
            use_ms_model=use_ms_model,
            model_path=emotion_model_path,
            use_gpu=use_gpu,
            overwrites=overwrites,
        )

    def analyze(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        transcript, detected_language = self.speech_recognizer.transcribe(audio_path)
        emotion_ranking = self.emotion_recognizer.predict_scores(audio_path)
        top_emotion = emotion_ranking[0]
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
        )

    @staticmethod
    def save_result(result, output_path):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

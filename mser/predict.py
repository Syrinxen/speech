from typing import List

from mser import DEFAULT_EMOTION2VEC_MODEL, SUPPORT_EMOTION2VEC_MODEL
from mser.utils.emotion2vec_predict import Emotion2vecPredict


class MSERPredictor:
    def __init__(self, emotion_model=None, use_gpu=True, log_level="info", **_kwargs):
        model_name = emotion_model or DEFAULT_EMOTION2VEC_MODEL
        assert model_name in SUPPORT_EMOTION2VEC_MODEL, f"不支持的 emotion2vec 模型：{model_name}"
        self.model_name = model_name
        self.predictor = Emotion2vecPredict(
            model_id=model_name,
            revision=None,
            use_gpu=use_gpu,
            log_level=log_level,
        )

    def predict(self, audio_data, sample_rate=16000):
        del sample_rate
        ranked_results = self.predict_scores(audio_data=audio_data)
        return ranked_results[0]["label"], ranked_results[0]["confidence"]

    def predict_scores(self, audio_data, sample_rate=16000):
        del sample_rate
        ranked_results = self.predictor.predict_scores(audio_data)
        return [
            {
                "label": label,
                "confidence": confidence,
                "score": round(float(confidence) * 100, 2),
            }
            for label, confidence in ranked_results[0]
        ]

    def predict_batch(self, audios_data: List, sample_rate=16000):
        del sample_rate
        ranked_batch = self.predictor.predict_scores(audios_data)
        labels, scores = [], []
        for ranked_results in ranked_batch:
            label, confidence = ranked_results[0]
            labels.append(label)
            scores.append(confidence)
        return labels, scores

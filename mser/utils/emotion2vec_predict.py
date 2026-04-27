import os
import shutil

from funasr import AutoModel
from loguru import logger
from modelscope import snapshot_download


class Emotion2vecPredict:
    def __init__(self, model_id, revision, use_gpu=True, log_level="info"):
        emotion2vec_model_dir = "models"
        save_model_dir = os.path.join(emotion2vec_model_dir, model_id)
        if not os.path.exists(save_model_dir):
            model_dir = snapshot_download(model_id, revision=revision)
            shutil.copytree(model_dir, save_model_dir)
        self.model = AutoModel(
            model=save_model_dir,
            log_level=log_level.upper(),
            device="cuda" if use_gpu else "cpu",
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
        )
        logger.info(f"成功加载 emotion2vec 模型：{save_model_dir}")

    def predict(self, audio):
        ranked_results = self.predict_scores(audio)
        labels, scores = [], []
        for result in ranked_results:
            label, score = result[0]
            labels.append(label)
            scores.append(score)
        return labels, scores

    def predict_scores(self, audio):
        res = self.model.generate(audio, granularity="utterance", extract_embedding=False)
        ranked_results = []
        for result in res:
            pairs = [
                (label.split("/")[0], round(float(score), 5))
                for label, score in zip(result["labels"], result["scores"])
            ]
            pairs.sort(key=lambda item: item[1], reverse=True)
            ranked_results.append(pairs)
        return ranked_results

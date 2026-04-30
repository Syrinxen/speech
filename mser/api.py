import os
import tempfile
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from mser import DEFAULT_EMOTION2VEC_MODEL
from mser.emotion_service import EmotionBusinessService
from mser.pipeline import SpeechEmotionPipeline


app = FastAPI(
    title="Emotion Echo API",
    version="0.1.0",
    description="面向情绪文本还原和情绪强度评估的多模态接口服务。",
)

business_service = EmotionBusinessService()


class AudioPathRequest(BaseModel):
    audio_path: str = Field(..., min_length=1, description="本地音频绝对路径")
    emotion_model: str = Field(DEFAULT_EMOTION2VEC_MODEL, description="emotion2vec 模型名称")
    whisper_model: str = Field("base", description="Whisper 模型大小")
    language: Optional[str] = Field(None, description="可选，强制指定语种，如 zh/en")
    use_gpu: bool = Field(True, description="是否启用 GPU")


class TextRestoreRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待还原文本")
    emotion: Optional[str] = Field(None, description="情绪标签，可为空")
    confidence: Optional[float] = Field(None, description="情绪置信度，范围 0-1")
    language: Optional[str] = Field(None, description="文本语言，可为空")


class IntensityRequest(BaseModel):
    emotion: str = Field(..., description="情绪标签")
    confidence: float = Field(..., ge=0.0, le=1.0, description="情绪置信度，范围 0-1")


@lru_cache(maxsize=8)
def get_pipeline(
    emotion_model: str,
    whisper_model: str,
    language: Optional[str],
    use_gpu: bool,
) -> SpeechEmotionPipeline:
    return SpeechEmotionPipeline(
        emotion_model=emotion_model,
        whisper_model=whisper_model,
        language=language,
        use_gpu=use_gpu,
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "emotion-echo-api"}


@app.post("/api/v1/emotion/analyze/audio-path")
def analyze_audio_path(request: AudioPathRequest):
    try:
        if not os.path.exists(request.audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {request.audio_path}")
        pipeline = get_pipeline(
            emotion_model=request.emotion_model,
            whisper_model=request.whisper_model,
            language=request.language,
            use_gpu=request.use_gpu,
        )
        result = pipeline.analyze(request.audio_path)
        return result.to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"audio analysis failed: {exc}") from exc


@app.post("/api/v1/emotion/analyze/upload")
async def analyze_audio_upload(
    file: UploadFile = File(...),
    emotion_model: str = Form(DEFAULT_EMOTION2VEC_MODEL),
    whisper_model: str = Form("base"),
    language: Optional[str] = Form(None),
    use_gpu: bool = Form(True),
):
    suffix = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            temp_file.write(await file.read())

        pipeline = get_pipeline(
            emotion_model=emotion_model,
            whisper_model=whisper_model,
            language=language,
            use_gpu=use_gpu,
        )
        result = pipeline.analyze(temp_path)
        payload = result.to_dict()
        payload["uploaded_filename"] = file.filename
        return payload
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"upload audio analysis failed: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/v1/emotion/restore-text")
def restore_text(request: TextRestoreRequest):
    return business_service.analyze_text(
        text=request.text,
        emotion=request.emotion,
        confidence=request.confidence,
        language=request.language,
    )


@app.post("/api/v1/emotion/evaluate-intensity")
def evaluate_intensity(request: IntensityRequest):
    return business_service.evaluate_intensity(
        emotion=request.emotion,
        confidence=request.confidence,
    ).to_dict()


def main():
    import uvicorn

    uvicorn.run("mser.api:app", host="0.0.0.0", port=8000)

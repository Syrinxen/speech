import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


SUPPORTED_EMOTIONS = {
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
}

EMOTION_DISPLAY_NAMES = {
    "angry": "愤怒",
    "disgust": "厌恶",
    "fear": "恐惧",
    "happy": "开心",
    "neutral": "平静",
    "sad": "悲伤",
    "surprise": "惊讶",
}

EMOTION_VALENCE = {
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "happy": "positive",
    "neutral": "neutral",
    "sad": "negative",
    "surprise": "mixed",
}

ZH_EMOTION_PREFIX = {
    "angry": "我现在真的有些生气",
    "disgust": "这件事让我有些抗拒",
    "fear": "我现在心里有些害怕",
    "happy": "我现在真的很开心",
    "neutral": "我想认真说明一下",
    "sad": "我现在心里挺难受的",
    "surprise": "这件事真的让我有些意外",
}

EN_EMOTION_PREFIX = {
    "angry": "I feel really angry",
    "disgust": "I feel uncomfortable about this",
    "fear": "I feel scared right now",
    "happy": "I feel really happy",
    "neutral": "I want to explain this calmly",
    "sad": "I feel very sad right now",
    "surprise": "I feel genuinely surprised",
}

INTENSITY_LEVELS = [
    (90.0, "very_high", "极强", "情绪表达非常强烈，建议优先关注触发事件和安抚策略。"),
    (75.0, "high", "高", "情绪比较强烈，已经明显影响表达方式和沟通节奏。"),
    (50.0, "medium", "中", "情绪较为明确，适合继续结合上下文做进一步干预。"),
    (25.0, "low", "低", "情绪存在但相对克制，适合继续观察后续变化。"),
    (0.0, "very_low", "很低", "当前情绪波动较小，整体表达相对平稳。"),
]

TEXT_EMOTION_KEYWORDS = {
    "angry": ["生气", "愤怒", "火大", "气死", "烦死", "恼火", "angry", "mad", "furious"],
    "disgust": ["恶心", "讨厌", "反感", "排斥", "厌恶", "disgust", "hate"],
    "fear": ["害怕", "担心", "恐惧", "紧张", "不安", "怕", "scared", "afraid", "fear", "anxious"],
    "happy": ["开心", "高兴", "快乐", "幸福", "满意", "激动", "happy", "glad", "excited", "delighted"],
    "sad": ["难过", "伤心", "委屈", "失落", "沮丧", "想哭", "sad", "upset", "depressed", "down"],
    "surprise": ["惊讶", "震惊", "意外", "没想到", "surprised", "shocked", "unexpected"],
}


@dataclass
class EmotionIntensity:
    score: float
    confidence: float
    level_code: str
    level_name: str
    description: str
    valence: str
    primary_emotion: str
    primary_emotion_name: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EmotionTextRestoration:
    normalized_text: str
    restored_text: str
    speaking_style: str
    suggestions: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def detect_language(text: str, preferred_language: Optional[str] = None) -> str:
    if preferred_language:
        return preferred_language
    return "zh" if _contains_chinese(text) else "en"


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def ensure_sentence_punctuation(text: str, language: str) -> str:
    if not text:
        return text
    if text[-1] in ".!?。！？":
        return text
    return f"{text}{'。' if language.startswith('zh') else '.'}"


def infer_text_emotion(text: str) -> Dict[str, float]:
    normalized = normalize_text(text).lower()
    if not normalized:
        return {"emotion": "neutral", "confidence": 0.5}

    score_by_emotion = {emotion: 0 for emotion in SUPPORTED_EMOTIONS}
    for emotion, keywords in TEXT_EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                score_by_emotion[emotion] += 1

    top_emotion = max(score_by_emotion, key=score_by_emotion.get)
    top_hits = score_by_emotion[top_emotion]
    if top_hits == 0:
        return {"emotion": "neutral", "confidence": 0.55}

    confidence = min(0.55 + top_hits * 0.12, 0.95)
    return {"emotion": top_emotion, "confidence": round(confidence, 4)}


class EmotionBusinessService:
    def evaluate_intensity(self, emotion: str, confidence: float) -> EmotionIntensity:
        normalized_emotion = emotion if emotion in SUPPORTED_EMOTIONS else "neutral"
        normalized_confidence = max(0.0, min(float(confidence), 1.0))
        score = round(normalized_confidence * 100, 2)

        for threshold, level_code, level_name, description in INTENSITY_LEVELS:
            if score >= threshold:
                return EmotionIntensity(
                    score=score,
                    confidence=normalized_confidence,
                    level_code=level_code,
                    level_name=level_name,
                    description=description,
                    valence=EMOTION_VALENCE.get(normalized_emotion, "neutral"),
                    primary_emotion=normalized_emotion,
                    primary_emotion_name=EMOTION_DISPLAY_NAMES.get(normalized_emotion, "平静"),
                )

        return EmotionIntensity(
            score=score,
            confidence=normalized_confidence,
            level_code="very_low",
            level_name="很低",
            description="当前情绪波动较小，整体表达相对平稳。",
            valence=EMOTION_VALENCE.get(normalized_emotion, "neutral"),
            primary_emotion=normalized_emotion,
            primary_emotion_name=EMOTION_DISPLAY_NAMES.get(normalized_emotion, "平静"),
        )

    def restore_text(
        self,
        text: str,
        emotion: str = "neutral",
        confidence: float = 0.5,
        language: Optional[str] = None,
    ) -> EmotionTextRestoration:
        normalized_text = normalize_text(text)
        detected_language = detect_language(normalized_text, language)
        intensity = self.evaluate_intensity(emotion=emotion, confidence=confidence)
        punctuated_text = ensure_sentence_punctuation(normalized_text, detected_language)

        if detected_language.startswith("zh"):
            prefix = ZH_EMOTION_PREFIX.get(intensity.primary_emotion, ZH_EMOTION_PREFIX["neutral"])
            separator = "，"
        else:
            prefix = EN_EMOTION_PREFIX.get(intensity.primary_emotion, EN_EMOTION_PREFIX["neutral"])
            separator = ", "

        if not punctuated_text:
            restored_text = ensure_sentence_punctuation(prefix, detected_language)
        elif intensity.score >= 50:
            restored_text = f"{prefix}{separator}{punctuated_text}"
        else:
            restored_text = punctuated_text

        speaking_style = self._build_speaking_style(intensity, detected_language)
        suggestions = self._build_suggestions(intensity, detected_language)
        return EmotionTextRestoration(
            normalized_text=punctuated_text,
            restored_text=restored_text,
            speaking_style=speaking_style,
            suggestions=suggestions,
        )

    def analyze_text(
        self,
        text: str,
        emotion: Optional[str] = None,
        confidence: Optional[float] = None,
        language: Optional[str] = None,
    ) -> Dict:
        inferred = None
        if emotion is None or confidence is None:
            inferred = infer_text_emotion(text)
        final_emotion = emotion or inferred["emotion"]
        final_confidence = confidence if confidence is not None else inferred["confidence"]
        intensity = self.evaluate_intensity(final_emotion, final_confidence)
        restoration = self.restore_text(
            text=text,
            emotion=final_emotion,
            confidence=final_confidence,
            language=language,
        )
        return {
            "emotion": final_emotion,
            "emotion_name": intensity.primary_emotion_name,
            "confidence": round(float(final_confidence), 4),
            "emotion_score": intensity.score,
            "intensity": intensity.to_dict(),
            "text_restoration": restoration.to_dict(),
            "inferred_from_text": inferred is not None,
        }

    def _build_speaking_style(self, intensity: EmotionIntensity, language: str) -> str:
        if language.startswith("zh"):
            return f"{intensity.primary_emotion_name}、{intensity.level_name}强度"
        return f"{intensity.primary_emotion} with {intensity.level_code} intensity"

    def _build_suggestions(self, intensity: EmotionIntensity, language: str) -> List[str]:
        if language.startswith("zh"):
            suggestions = {
                "angry": [
                    "建议先描述引发愤怒的具体事件，再表达自己的核心诉求。",
                    "如果用于心理疏导，可继续追问愤怒背后的失望或受伤感。",
                ],
                "sad": [
                    "建议补充最近一次触发低落情绪的时间点和场景。",
                    "如果用于对话干预，可优先给出共情式回应，避免直接讲道理。",
                ],
                "fear": [
                    "建议继续确认担忧对象、最坏预期和现实证据。",
                    "如果用于疏导，可先降低不确定感，再进入问题分析。",
                ],
                "happy": [
                    "建议记录带来积极情绪的事件，便于后续建立正向经验库。",
                    "如果用于陪伴式对话，可鼓励继续展开具体细节。",
                ],
            }
            return suggestions.get(
                intensity.primary_emotion,
                [
                    "建议结合上下文继续收集事件、对象和时间线信息。",
                    "如果进入心理分析，可补充最近一次相同情绪出现的场景。",
                ],
            )

        suggestions = {
            "angry": [
                "Describe the trigger first, then state the unmet need clearly.",
                "For counseling, explore whether anger is covering hurt or disappointment.",
            ],
            "sad": [
                "Add the latest triggering event and the main loss behind it.",
                "Use empathic reflection before moving into problem solving.",
            ],
            "fear": [
                "Clarify the feared outcome and the evidence supporting it.",
                "Reduce uncertainty first, then discuss coping options.",
            ],
        }
        return suggestions.get(
            intensity.primary_emotion,
            [
                "Collect more context about the event, people involved, and timeline.",
                "Ask for one recent example to make the emotional signal easier to interpret.",
            ],
        )

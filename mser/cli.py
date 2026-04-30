import argparse
import json

from mser import DEFAULT_EMOTION2VEC_MODEL, SUPPORT_EMOTION2VEC_MODEL
from mser.pipeline import SpeechEmotionPipeline


def build_parser():
    parser = argparse.ArgumentParser(
        description="Speech-to-text and emotion recognition with emotion2vec."
    )
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file.")
    parser.add_argument(
        "--emotion_model",
        type=str,
        default=DEFAULT_EMOTION2VEC_MODEL,
        choices=SUPPORT_EMOTION2VEC_MODEL,
        help="emotion2vec model name.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        help="Whisper model size, such as tiny/base/small/medium/large.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force transcription language, for example zh or en.",
    )
    parser.add_argument(
        "--use_gpu",
        type=lambda x: str(x).lower() in {"true", "1", "yes", "y"},
        default=True,
        help="Whether to use GPU.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def format_console_output(result):
    ranking_preview = ", ".join(
        f"{item.label}:{item.score:.2f}" for item in result.emotion_ranking[:3]
    )
    return "\n".join(
        [
            f"audio_path: {result.audio_path}",
            f"transcript: {result.transcript}",
            f"detected_language: {result.detected_language}",
            f"emotion: {result.emotion}",
            f"confidence: {result.confidence:.5f}",
            f"emotion_score: {result.emotion_score:.2f}",
            f"intensity_level: {result.intensity.level_name} ({result.intensity.level_code})",
            f"emotion_description: {result.intensity.description}",
            f"restored_text: {result.text_restoration.restored_text}",
            f"top3_emotions: {ranking_preview}",
        ]
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    pipeline = SpeechEmotionPipeline(
        emotion_model=args.emotion_model,
        whisper_model=args.whisper_model,
        language=args.language,
        use_gpu=args.use_gpu,
    )
    result = pipeline.analyze(args.audio_path)
    print(format_console_output(result))
    if args.output_path:
        pipeline.save_result(result, args.output_path)
        print(f"json_saved_to: {args.output_path}")
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

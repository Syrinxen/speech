import argparse
import json

from mser.pipeline import SpeechEmotionPipeline


def build_parser():
    parser = argparse.ArgumentParser(
        description="Speech-to-text and emotion recognition pipeline."
    )
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file.")
    parser.add_argument(
        "--configs",
        type=str,
        default="configs/bi_lstm.yml",
        help="Emotion recognition config file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/BiLSTM_Emotion2Vec/best_model/",
        help="Local emotion recognition model path.",
    )
    parser.add_argument(
        "--use_ms_model",
        type=str,
        default=None,
        help="Optional ModelScope emotion2vec model name.",
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
        "--overwrites",
        type=str,
        default=None,
        help="Override config values, for example train_conf.max_epoch=1.",
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
            f"top3_emotions: {ranking_preview}",
        ]
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    pipeline = SpeechEmotionPipeline(
        emotion_configs=args.configs,
        emotion_model_path=args.model_path,
        use_ms_model=args.use_ms_model,
        whisper_model=args.whisper_model,
        language=args.language,
        use_gpu=args.use_gpu,
        overwrites=args.overwrites,
    )
    result = pipeline.analyze(args.audio_path)
    print(format_console_output(result))
    if args.output_path:
        pipeline.save_result(result, args.output_path)
        print(f"json_saved_to: {args.output_path}")
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

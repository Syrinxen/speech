import argparse

from mser import DEFAULT_EMOTION2VEC_MODEL, SUPPORT_EMOTION2VEC_MODEL
from mser.predict import MSERPredictor


def main():
    parser = argparse.ArgumentParser(description="Emotion recognition with emotion2vec.")
    parser.add_argument("--audio_path", type=str, default="dataset/test.wav", help="音频路径")
    parser.add_argument(
        "--emotion_model",
        type=str,
        default=DEFAULT_EMOTION2VEC_MODEL,
        choices=SUPPORT_EMOTION2VEC_MODEL,
        help="emotion2vec 模型名称",
    )
    parser.add_argument(
        "--use_gpu",
        type=lambda x: str(x).lower() in {"true", "1", "yes", "y"},
        default=True,
        help="是否使用 GPU",
    )
    args = parser.parse_args()

    predictor = MSERPredictor(emotion_model=args.emotion_model, use_gpu=args.use_gpu)
    label, confidence = predictor.predict(audio_data=args.audio_path)
    print(f"audio_path: {args.audio_path}")
    print(f"emotion: {label}")
    print(f"confidence: {confidence:.5f}")


if __name__ == "__main__":
    main()

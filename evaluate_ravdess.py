import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from mser import DEFAULT_EMOTION2VEC_MODEL
from mser.predict import MSERPredictor


RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

RAVDESS_INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}

MODEL_LABEL_MAP = {
    "angry": "angry",
    "生气": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "厌恶": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "恐惧": "fear",
    "happy": "happy",
    "开心": "happy",
    "neutral": "neutral",
    "中立": "neutral",
    "sad": "sad",
    "难过": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "惊讶": "surprise",
    "吃惊": "surprise",
    "other": "other",
    "其他": "other",
    "unknown": "unknown",
    "未知": "unknown",
    "calm": "calm",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate emotion2vec on RAVDESS.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="Audio_Speech_Actors_01-24",
        help="Path to the RAVDESS Audio_Speech_Actors_01-24 directory.",
    )
    parser.add_argument(
        "--emotion_model",
        type=str,
        default=DEFAULT_EMOTION2VEC_MODEL,
        help="emotion2vec model name.",
    )
    parser.add_argument(
        "--use_gpu",
        type=lambda x: str(x).lower() in {"true", "1", "yes", "y"},
        default=False,
        help="Whether to use GPU.",
    )
    parser.add_argument(
        "--merge_calm_to_neutral",
        type=lambda x: str(x).lower() in {"true", "1", "yes", "y"},
        default=False,
        help="Whether to merge RAVDESS calm into neutral. Default is False and calm samples are skipped.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/ravdess_eval",
        help="Directory to save evaluation artifacts.",
    )
    return parser.parse_args()


def parse_ravdess_file(path: Path) -> Optional[Dict[str, str]]:
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    emotion = RAVDESS_EMOTION_MAP.get(parts[2])
    intensity = RAVDESS_INTENSITY_MAP.get(parts[3], "unknown")
    if emotion is None:
        return None
    return {
        "emotion": emotion,
        "intensity": intensity,
        "actor": parts[6],
        "statement": parts[4],
        "repetition": parts[5],
    }


def normalize_predicted_label(label: str) -> str:
    normalized = MODEL_LABEL_MAP.get(label, label)
    return normalized


def resolve_ground_truth_label(label: str, merge_calm_to_neutral: bool) -> Optional[str]:
    if label == "calm":
        return "neutral" if merge_calm_to_neutral else None
    return label


def collect_files(dataset_dir: Path, max_samples: Optional[int]) -> List[Path]:
    files = sorted(dataset_dir.rglob("*.wav"))
    if max_samples is not None:
        files = files[:max_samples]
    return files


def ensure_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def save_predictions_csv(output_path: Path, rows: List[Dict[str, object]]):
    fieldnames = [
        "file_path",
        "actor",
        "statement",
        "repetition",
        "intensity",
        "ground_truth",
        "prediction",
        "top1_confidence",
        "top3",
        "is_correct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_csv(output_path: Path, labels: List[str], matrix):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth\\prediction", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def evaluate(args):
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    predictor = MSERPredictor(emotion_model=args.emotion_model, use_gpu=args.use_gpu)
    files = collect_files(dataset_dir, args.max_samples)

    y_true: List[str] = []
    y_pred: List[str] = []
    prediction_rows: List[Dict[str, object]] = []
    skipped_calm = 0
    skipped_invalid = 0

    for index, audio_path in enumerate(files, start=1):
        parsed = parse_ravdess_file(audio_path)
        if parsed is None:
            skipped_invalid += 1
            continue

        ground_truth = resolve_ground_truth_label(parsed["emotion"], args.merge_calm_to_neutral)
        if ground_truth is None:
            skipped_calm += 1
            continue

        ranked = predictor.predict_scores(str(audio_path))
        if not ranked:
            continue

        predicted_label = normalize_predicted_label(ranked[0]["label"])
        top1_confidence = ranked[0]["confidence"]
        top3 = ", ".join(
            f"{normalize_predicted_label(item['label'])}:{item['confidence']:.5f}"
            for item in ranked[:3]
        )

        y_true.append(ground_truth)
        y_pred.append(predicted_label)
        prediction_rows.append(
            {
                "file_path": str(audio_path),
                "actor": parsed["actor"],
                "statement": parsed["statement"],
                "repetition": parsed["repetition"],
                "intensity": parsed["intensity"],
                "ground_truth": ground_truth,
                "prediction": predicted_label,
                "top1_confidence": round(float(top1_confidence), 5),
                "top3": top3,
                "is_correct": ground_truth == predicted_label,
            }
        )

        if index % 50 == 0 or index == len(files):
            print(f"processed {index}/{len(files)} files")

    if not y_true:
        raise RuntimeError("No valid evaluation samples were collected.")

    base_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    observed_extra_labels = sorted({label for label in y_pred if label not in base_labels})
    labels_for_report = base_labels + observed_extra_labels

    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels_for_report, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels_for_report, average="weighted", zero_division=0)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_for_report,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels_for_report)

    intensity_groups = {}
    for group in ("normal", "strong"):
        group_true = [row["ground_truth"] for row in prediction_rows if row["intensity"] == group]
        group_pred = [row["prediction"] for row in prediction_rows if row["intensity"] == group]
        if group_true:
            intensity_groups[group] = {
                "samples": len(group_true),
                "accuracy": round(accuracy_score(group_true, group_pred), 6),
                "macro_f1": round(
                    f1_score(group_true, group_pred, labels=labels_for_report, average="macro", zero_division=0),
                    6,
                ),
            }

    summary = {
        "dataset_dir": str(dataset_dir.resolve()),
        "emotion_model": args.emotion_model,
        "use_gpu": args.use_gpu,
        "merge_calm_to_neutral": args.merge_calm_to_neutral,
        "total_wav_files": len(files),
        "evaluated_samples": len(y_true),
        "skipped_calm_samples": skipped_calm,
        "skipped_invalid_files": skipped_invalid,
        "ground_truth_distribution": dict(Counter(y_true)),
        "prediction_distribution": dict(Counter(y_pred)),
        "overall_accuracy": round(float(overall_accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "intensity_groups": intensity_groups,
        "labels_for_report": labels_for_report,
        "classification_report": report_dict,
    }

    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.csv"
    confusion_path = output_dir / "confusion_matrix.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    save_predictions_csv(predictions_path, prediction_rows)
    save_confusion_matrix_csv(confusion_path, labels_for_report, conf_matrix)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_saved_to: {summary_path}")
    print(f"predictions_saved_to: {predictions_path}")
    print(f"confusion_matrix_saved_to: {confusion_path}")


if __name__ == "__main__":
    evaluate(parse_args())

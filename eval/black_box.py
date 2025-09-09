import json
import argparse
import numpy as np
from pathlib import Path
import re


def black_box_evaluation(prediction_path, json_path):
    """
    Calculates and prints black-box evaluation metrics (Acc, OBOA, MAE, RMSE).
    """
    try:
        with open(json_path, "r") as f:
            gt_file = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {json_path}")
        return

    gt_data = {str(item["index"]): int(item["answer"]) for item in gt_file}

    prediction_files = list(Path(prediction_path).glob("*.txt"))
    if not prediction_files:
        print(f"Error: No prediction files (.txt) found in {prediction_path}")
        return

    preds = {}
    for file_path in prediction_files:
        with open(file_path, "r") as f:
            try:
                prediction_id = file_path.stem
                content = f.read().strip()

                match = re.search(r"-?\d+", content)
                if match:
                    preds[prediction_id] = int(match.group(0))
                else:
                    preds[prediction_id] = 0
            except (ValueError, IndexError):
                preds[file_path.stem] = 0

    acc, oboa, mae, count = 0, 0, 0, 0
    sq_diffs = []

    for pred_id, pred_count in preds.items():
        if pred_id in gt_data:
            gt_count = gt_data[pred_id]

            if pred_count == gt_count:
                acc += 1
            if abs(pred_count - gt_count) <= 1:
                oboa += 1

            mae += abs(pred_count - gt_count)
            sq_diffs.append((pred_count - gt_count) ** 2)
            count += 1

    if count > 0:
        print(f"\n--- Evaluation Results for: {prediction_path} ---")
        print(f"Total predictions matched with ground truth: {count}")
        print(f"Accuracy (Acc): {acc/count:.4f}")
        print(f"Off-By-One Accuracy (OBOA): {oboa/count:.4f}")
        print(f"Mean Absolute Error (MAE): {mae/count:.4f}")
        print(f"Root Mean Square Error (RMSE): {np.sqrt(np.mean(sq_diffs)):.4f}")
        print(f"--------------------------------------------------")
    else:
        print("No valid predictions could be matched with the ground truth.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Black-Box evaluation for CG-AV-Counting."
    )
    parser.add_argument(
        "prediction_path",
        type=str,
        help="Directory containing the model's prediction text files.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/scratch/ykulka10/data/DATA_ROOT/CG-AV-Counting/cg-av-counting.json",
        help="Path to the ground truth dataset JSON file.",
    )
    args = parser.parse_args()
    black_box_evaluation(args.prediction_path, args.json_path)

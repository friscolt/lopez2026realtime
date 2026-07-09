"""
Aggregate per-seed metrics (mean +/- std across the 3 seeds, for each of the 3 views) into a single
summary CSV/JSON. Replaces the 12 summarize_{baseline_results,tta,roi_results,roi_tta_results}.py
scripts (one per backbone x {roi, tta} combination).

Example:
  python scripts/summarize_results.py --backbone resnet34 --roi --tta
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")

VIEWS = ["SUR", "SEC", "MIX"]
SEEDS = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["resnet34", "resnet50", "vit_small"])
    parser.add_argument("--roi", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    args = parser.parse_args()

    model_dir_name = f"{args.backbone}_roi_baseline" if args.roi else f"{args.backbone}_baseline"
    model_dir = f"{args.data_root}/models/{model_dir_name}"

    metrics_suffix = "tta_metrics" if args.tta else "metrics"
    file_prefix = f"{args.backbone}_roi" if args.roi else args.backbone

    summary_name = "_".join(filter(None, [args.backbone, "roi" if args.roi else None, "tta" if args.tta else None, "summary"]))
    output_csv = os.path.join(model_dir, f"{summary_name}.csv")
    output_json = os.path.join(model_dir, f"{summary_name}.json")

    results = []

    for view in VIEWS:
        acc_list, precision_list, recall_list, f1_list = [], [], [], []

        for seed in SEEDS:
            path = f"{model_dir}/{file_prefix}_{view}_seed{seed}_{metrics_suffix}.json"

            if not os.path.exists(path):
                print("Missing:", path)
                continue

            with open(path) as f:
                data = json.load(f)

            acc_list.append(data["accuracy"])
            precision_list.append(data["weighted avg"]["precision"])
            recall_list.append(data["weighted avg"]["recall"])
            f1_list.append(data["weighted avg"]["f1-score"])

        results.append({
            "dataset": view,
            "accuracy_mean": np.mean(acc_list),
            "accuracy_std": np.std(acc_list),
            "precision_mean": np.mean(precision_list),
            "precision_std": np.std(precision_list),
            "recall_mean": np.mean(recall_list),
            "recall_std": np.std(recall_list),
            "f1_mean": np.mean(f1_list),
            "f1_std": np.std(f1_list),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print("\nSummary saved:")
    print(output_csv)
    print(output_json)
    print("\nResults:")
    print(df)


if __name__ == "__main__":
    main()

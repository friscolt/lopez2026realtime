"""
Print a results table (plain + LaTeX) aggregating ProtoNet eval_results[.txt/_tta.txt] across
views/shots for the full-image and ROI experiments. Replaces eval_results_fsl.py +
summarize_results_fsl_tta.py.

Example:
  python scripts/summarize_fewshot_results.py --tta
"""
import argparse
import os
import re

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")

EXPERIMENTS = {
    "Full": "protonet_resnet34_full/protonet_resnet34",
    "ROI": "protonet_resnet34_roi/protonet_resnet34",
}
VIEWS = ["SUR", "SEC", "MIX"]
SHOTS = [1, 3, 5]


def read_result(file_path):
    if not os.path.exists(file_path):
        return None

    with open(file_path) as f:
        content = f.read()

    match = re.search(r"Accuracy.*:\s*([0-9.]+)\s*±\s*([0-9.]+)", content)
    if match:
        return float(match.group(1)), float(match.group(2))

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    args = parser.parse_args()

    base_dir = f"{args.data_root}/models_fsl"
    result_filename = "eval_results_tta.txt" if args.tta else "eval_results.txt"

    results = {}
    for data_type, path in EXPERIMENTS.items():
        results[data_type] = {}
        for view in VIEWS:
            results[data_type][view] = {}
            for shot in SHOTS:
                file_path = os.path.join(base_dir, path, view, f"{shot}shot", result_filename)
                res = read_result(file_path)
                results[data_type][view][shot] = f"{res[0]:.3f} ± {res[1]:.3f}" if res else "N/A"

    title = "TTA RESULTS TABLE" if args.tta else "RESULTS TABLE"
    print(f"\n================ {title} ================\n")

    header = f"{'Data':<6} {'View':<5} {'1-shot':<15} {'3-shot':<15} {'5-shot':<15}"
    print(header)
    print("-" * len(header))

    for data_type in results:
        for view in VIEWS:
            row = f"{data_type:<6} {view:<5} "
            for shot in SHOTS:
                row += f"{results[data_type][view][shot]:<15} "
            print(row)

    print("\n================ LATEX TABLE ================\n")
    print("\\begin{tabular}{l l c c c}")
    print("\\hline")
    print("Data & View & 1-shot & 3-shot & 5-shot \\\\")
    print("\\hline")

    for data_type in results:
        for view in VIEWS:
            row = f"{data_type} & {view}"
            for shot in SHOTS:
                row += f" & {results[data_type][view][shot]}"
            row += " \\\\"
            print(row)

    print("\\hline")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()

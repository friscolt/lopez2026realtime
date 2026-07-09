import os
import re

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

BASE_DIR = f"{DATA_ROOT}/models_fsl"

EXPERIMENTS = {
    "Full": "protonet_resnet34_full/protonet_resnet34",
    "ROI": "protonet_resnet34_roi/protonet_resnet34"
}

VIEWS = ["SUR", "SEC", "MIX"]
SHOTS = [1, 3, 5]


def read_tta_result(file_path):
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        content = f.read()

    match = re.search(r"Accuracy.*:\s*([0-9.]+)\s*±\s*([0-9.]+)", content)
    if match:
        mean = float(match.group(1))
        std = float(match.group(2))
        return mean, std

    return None


def main():
    results = {}

    for data_type, path in EXPERIMENTS.items():
        results[data_type] = {}

        for view in VIEWS:
            results[data_type][view] = {}

            for shot in SHOTS:

                file_path = os.path.join(
                    BASE_DIR,
                    path,
                    view,
                    f"{shot}shot",
                    "eval_results_tta.txt"
                )

                res = read_tta_result(file_path)

                if res:
                    mean, std = res
                    results[data_type][view][shot] = f"{mean:.3f} ± {std:.3f}"
                else:
                    results[data_type][view][shot] = "N/A"

    # ===== PRINT TABLE =====
    print("\n================ TTA RESULTS TABLE ================\n")

    header = f"{'Data':<6} {'View':<5} {'1-shot':<15} {'3-shot':<15} {'5-shot':<15}"
    print(header)
    print("-" * len(header))

    for data_type in results:
        for view in VIEWS:
            row = f"{data_type:<6} {view:<5} "
            for shot in SHOTS:
                row += f"{results[data_type][view][shot]:<15} "
            print(row)

    # ===== LATEX TABLE =====
    print("\n================ LATEX TABLE (TTA) ================\n")

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
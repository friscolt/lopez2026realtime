import os
import json
import pandas as pd
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# ===============================
# CONFIG
# ===============================

MODEL_DIR = f"{DATA_ROOT}/models/vit_small_baseline"

VIEWS = ["SUR", "SEC", "MIX"]
SEEDS = [0,1,2]

OUTPUT_CSV = os.path.join(MODEL_DIR,"vit_small_summary.csv")
OUTPUT_JSON = os.path.join(MODEL_DIR,"vit_small_summary.json")

results = []

# ===============================
# READ RESULTS
# ===============================

for view in VIEWS:

    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for seed in SEEDS:

        path = f"{MODEL_DIR}/vit_small_{view}_seed{seed}_metrics.json"

        if not os.path.exists(path):
            print("Missing:", path)
            continue

        with open(path,"r") as f:
            data = json.load(f)

        acc_list.append(data["accuracy"])
        precision_list.append(data["weighted avg"]["precision"])
        recall_list.append(data["weighted avg"]["recall"])
        f1_list.append(data["weighted avg"]["f1-score"])

    results.append({

        "dataset":view,

        "accuracy_mean":np.mean(acc_list),
        "accuracy_std":np.std(acc_list),

        "precision_mean":np.mean(precision_list),
        "precision_std":np.std(precision_list),

        "recall_mean":np.mean(recall_list),
        "recall_std":np.std(recall_list),

        "f1_mean":np.mean(f1_list),
        "f1_std":np.std(f1_list)

    })

# ===============================
# SAVE RESULTS
# ===============================

df = pd.DataFrame(results)

df.to_csv(OUTPUT_CSV,index=False)

with open(OUTPUT_JSON,"w") as f:
    json.dump(results,f,indent=4)

print("\nSummary saved:")
print(OUTPUT_CSV)
print(OUTPUT_JSON)

print("\nResults:")
print(df)
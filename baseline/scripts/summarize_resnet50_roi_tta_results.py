import os
import json
import pandas as pd
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

MODEL_DIR=f"{DATA_ROOT}/models/resnet50_roi_baseline"

VIEWS=["SUR","SEC","MIX"]
SEEDS=[0,1,2]

OUTPUT_CSV=os.path.join(MODEL_DIR,"resnet50_roi_tta_summary.csv")
OUTPUT_JSON=os.path.join(MODEL_DIR,"resnet50_roi_tta_summary.json")

results=[]

for view in VIEWS:

    acc=[]
    precision=[]
    recall=[]
    f1=[]

    for seed in SEEDS:

        path=f"{MODEL_DIR}/resnet50_roi_{view}_seed{seed}_tta_metrics.json"

        if not os.path.exists(path):
            print("Missing:",path)
            continue

        with open(path) as f:
            data=json.load(f)

        acc.append(data["accuracy"])
        precision.append(data["weighted avg"]["precision"])
        recall.append(data["weighted avg"]["recall"])
        f1.append(data["weighted avg"]["f1-score"])

    results.append({
        "dataset":view,
        "accuracy_mean":np.mean(acc),
        "accuracy_std":np.std(acc),
        "precision_mean":np.mean(precision),
        "precision_std":np.std(precision),
        "recall_mean":np.mean(recall),
        "recall_std":np.std(recall),
        "f1_mean":np.mean(f1),
        "f1_std":np.std(f1)
    })

df=pd.DataFrame(results)

df.to_csv(OUTPUT_CSV,index=False)

with open(OUTPUT_JSON,"w") as f:
    json.dump(results,f,indent=4)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_JSON)
print("\nResults:\n",df)
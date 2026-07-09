import os
import cv2
import pandas as pd

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

DATASETS = ["SUR","SEC","MIX"]
SPLITS = ["train","test"]

ROOT = f"{DATA_ROOT}/roi_dataset"

results = []

for dataset in DATASETS:
    
    for split in SPLITS:
        
        path = os.path.join(ROOT,dataset,split)
        
        classes = os.listdir(path)
        
        for cls in classes:
            
            class_path = os.path.join(path,cls)
            
            images = [
                f for f in os.listdir(class_path)
                if f.endswith((".png",".jpg",".jpeg"))
            ]
            
            for img_name in images:
                
                img_path = os.path.join(class_path,img_name)
                
                img = cv2.imread(img_path)
                
                h,w,_ = img.shape
                
                area = h*w
                
                results.append({
                    
                    "dataset":dataset,
                    "split":split,
                    "class":cls,
                    "width":w,
                    "height":h,
                    "area":area
                })

df = pd.DataFrame(results)

summary = df.groupby("dataset").agg({
    
    "width":["mean","min","max"],
    "height":["mean","min","max"],
    "area":["mean","min","max"]

})

print(summary)

df.to_csv(f"{DATA_ROOT}/roi_dataset/roi_sizes.csv",index=False)
summary.to_csv(f"{DATA_ROOT}/roi_dataset/roi_size_summary.csv")

print("\nSaved:")
print(f"{DATA_ROOT}/roi_dataset/roi_sizes.csv")
print(f"{DATA_ROOT}/roi_dataset/roi_size_summary.csv")
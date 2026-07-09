import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

parser = argparse.ArgumentParser()

parser.add_argument("--view",required=True,choices=["SUR","SEC","MIX"])
parser.add_argument("--seed",type=int,default=0)

args=parser.parse_args()

VIEW=args.view
SEED=args.seed

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

VIEW_ROOT = f"{DATA_ROOT}/roi_dataset_v4/{VIEW}"

TRAIN_DIR=os.path.join(VIEW_ROOT,"train")
TEST_DIR=os.path.join(VIEW_ROOT,"test")

MODEL_DIR=f"{DATA_ROOT}/models/vit_small_roi_baseline"
os.makedirs(MODEL_DIR,exist_ok=True)

MODEL_PATH=f"{MODEL_DIR}/vit_small_roi_{VIEW}_seed{SEED}_best.pth"

CSV_PATH=f"{MODEL_DIR}/vit_small_roi_{VIEW}_seed{SEED}_metrics.csv"
JSON_PATH=f"{MODEL_DIR}/vit_small_roi_{VIEW}_seed{SEED}_metrics.json"

IMG_SIZE=224
BATCH_SIZE=16
EPOCHS=40
LR=3e-5
PATIENCE=7

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

train_transform=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.05,
        hue=0.02
    ),
    transforms.ToTensor()
])

test_transform=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])

train_dataset=datasets.ImageFolder(TRAIN_DIR,transform=train_transform)
test_dataset=datasets.ImageFolder(TEST_DIR,transform=test_transform)

train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)

NUM_CLASSES=len(train_dataset.classes)

print("Classes:",train_dataset.classes)

model=timm.create_model(
    "vit_small_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model=model.to(DEVICE)

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)

scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

best_loss=float("inf")
counter=0

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()

    train_loss=0

    for images,labels in tqdm(train_loader):

        images=images.to(DEVICE)
        labels=labels.to(DEVICE)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        train_loss+=loss.item()

    train_loss/=len(train_loader)

    model.eval()

    val_loss=0

    y_true=[]
    y_pred=[]

    with torch.no_grad():

        for images,labels in test_loader:

            images=images.to(DEVICE)
            labels=labels.to(DEVICE)

            outputs=model(images)

            loss=criterion(outputs,labels)

            val_loss+=loss.item()

            _,pred=torch.max(outputs,1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    val_loss/=len(test_loader)

    print("Train Loss:",train_loss)
    print("Val Loss:",val_loss)

    scheduler.step(val_loss)

    if val_loss < best_loss:

        best_loss=val_loss
        counter=0

        torch.save(model.state_dict(),MODEL_PATH)

        print("Model saved")

    else:

        counter+=1

        if counter>=PATIENCE:
            print("Early stopping")
            break

model.load_state_dict(torch.load(MODEL_PATH))

model.eval()

y_true=[]
y_pred=[]

with torch.no_grad():

    for images,labels in test_loader:

        images=images.to(DEVICE)

        outputs=model(images)

        _,pred=torch.max(outputs,1)

        y_true.extend(labels.numpy())
        y_pred.extend(pred.cpu().numpy())

report=classification_report(
    y_true,
    y_pred,
    target_names=train_dataset.classes,
    output_dict=True
)

df=pd.DataFrame(report).transpose()

df.to_csv(CSV_PATH)

with open(JSON_PATH,"w") as f:
    json.dump(report,f,indent=4)

print("Metrics saved")
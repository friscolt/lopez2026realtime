import os
import cv2

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

ROOT = f"{DATA_ROOT}/SUR"

converted = 0

for root, dirs, files in os.walk(ROOT):

    if "mask" in root:
        continue

    for f in files:

        if f.lower().endswith((".jpg",".jpeg")):

            path = os.path.join(root, f)

            img = cv2.imread(path)

            new_name = os.path.splitext(f)[0] + ".png"
            new_path = os.path.join(root, new_name)

            cv2.imwrite(new_path, img)

            os.remove(path)

            converted += 1

print("Converted images:", converted)
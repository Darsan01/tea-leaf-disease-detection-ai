import os
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = r"D:\tealef\tealeafnet\5000_tea_leaf_with_blackbg_geotagged"

class_full_names = {
    "BB": "Brown Blight",
    "GL": "Healthy Leaf",
    "RR": "Red Rust",
    "RSM": "Red Spider Mite"
}

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

plt.figure(figsize=(10, 8))

image_number = 1

for class_folder in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_folder)

    if os.path.isdir(class_path):
        images = [
            file for file in os.listdir(class_path)
            if file.lower().endswith(image_extensions)
        ]

        if len(images) > 0:
            image_path = os.path.join(class_path, images[0])
            img = Image.open(image_path)

            plt.subplot(2, 2, image_number)
            plt.imshow(img)
            plt.title(f"{class_folder} - {class_full_names[class_folder]}")
            plt.axis("off")

            image_number += 1

plt.suptitle("Sample Images from TeaLeafNet Dataset")
plt.tight_layout()
plt.show()
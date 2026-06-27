import os

DATASET_PATH = r"D:\tealef\tealeafnet\5000_tea_leaf_with_blackbg_geotagged"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("Dataset path:")
print(DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    print("Dataset folder not found.")
    exit()

print("\nClass folders and image counts:")

total_images = 0

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)

    if os.path.isdir(class_path):
        image_count = 0

        for file in os.listdir(class_path):
            if file.lower().endswith(image_extensions):
                image_count += 1

        total_images += image_count
        print(f"{class_name}: {image_count} images")

print("\nTotal images:", total_images)
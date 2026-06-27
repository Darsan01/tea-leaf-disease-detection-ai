import os
import json
import numpy as np
from PIL import Image

BASE_DIR = "/mnt/d/tealef"

ORIGINAL_DATASET_DIR = os.path.join(
    BASE_DIR,
    "tealeafnet",
    "5000_tea_leaf_with_blackbg_geotagged"
)

ADMIN_TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")

OUTPUT_FILE = os.path.join(BASE_DIR, "known_image_hashes.json")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def average_hash(image, hash_size=16):
    image = image.convert("L").resize((hash_size, hash_size))
    arr = np.array(image)
    avg = arr.mean()

    bits = []

    for pixel in arr.flatten():
        bits.append("1" if pixel > avg else "0")

    return "".join(bits)


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def collect_hashes_from_folder(base_folder, source_name):
    known_images = []
    class_counts = {}

    if not os.path.exists(base_folder):
        print("Folder not found:", base_folder)
        return known_images, class_counts

    for root, dirs, files in os.walk(base_folder):
        for filename in files:
            if not is_image_file(filename):
                continue

            image_path = os.path.join(root, filename)
            class_code = os.path.basename(os.path.dirname(image_path))

            try:
                image = Image.open(image_path).convert("RGB")
                image_hash = average_hash(image)

                known_images.append({
                    "hash": image_hash,
                    "class_code": class_code,
                    "filename": filename,
                    "path": image_path,
                    "source": source_name
                })

                class_counts[class_code] = class_counts.get(class_code, 0) + 1

            except Exception as e:
                print("Skipped:", image_path, e)

    return known_images, class_counts


def merge_counts(counts_1, counts_2):
    final_counts = {}

    for key, value in counts_1.items():
        final_counts[key] = final_counts.get(key, 0) + value

    for key, value in counts_2.items():
        final_counts[key] = final_counts.get(key, 0) + value

    return final_counts


def main():
    print("Building known image hash database...")

    original_hashes, original_counts = collect_hashes_from_folder(
        ORIGINAL_DATASET_DIR,
        "original_dataset"
    )

    admin_hashes, admin_counts = collect_hashes_from_folder(
        ADMIN_TRAINING_DATA_DIR,
        "admin_training_data"
    )

    all_hashes = original_hashes + admin_hashes
    final_counts = merge_counts(original_counts, admin_counts)

    with open(OUTPUT_FILE, "w") as file:
        json.dump(all_hashes, file, indent=4)

    print("Known image hash database created successfully.")
    print("Saved to:", OUTPUT_FILE)
    print("Original dataset images:", len(original_hashes))
    print("Admin training images:", len(admin_hashes))
    print("Total known images:", len(all_hashes))
    print("Class counts:", final_counts)


if __name__ == "__main__":
    main()
import os
import json
import random
import shutil
from datetime import datetime

# CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf

BASE_DIR = "/mnt/d/tealef"

ORIGINAL_DATASET_DIR = os.path.join(
    BASE_DIR,
    "tealeafnet",
    "5000_tea_leaf_with_blackbg_geotagged"
)

ADMIN_TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.json")

MODEL_VERSIONS_DIR = os.path.join(BASE_DIR, "model_versions")
RETRAIN_LOG_DIR = os.path.join(BASE_DIR, "retrain_logs")

ACTIVE_MODEL_PATH = os.path.join(MODEL_VERSIONS_DIR, "active_model.keras")
ACTIVE_CLASSES_PATH = os.path.join(MODEL_VERSIONS_DIR, "active_classes.json")
LATEST_REPORT_PATH = os.path.join(RETRAIN_LOG_DIR, "latest_retrain_report.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_CLASSES = {
    "BB": "Brown Blight",
    "GL": "Healthy Leaf",
    "RR": "Red Rust",
    "RSM": "Red Spider Mite"
}

os.makedirs(MODEL_VERSIONS_DIR, exist_ok=True)
os.makedirs(RETRAIN_LOG_DIR, exist_ok=True)


def load_classes():
    if not os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "w") as file:
            json.dump(DEFAULT_CLASSES, file, indent=4)
        return DEFAULT_CLASSES

    with open(CLASSES_FILE, "r") as file:
        classes = json.load(file)

    if not classes:
        return DEFAULT_CLASSES

    return classes


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def collect_images_from_folder(base_folder, class_codes):
    image_paths = []
    labels = []

    if not os.path.exists(base_folder):
        return image_paths, labels

    for class_code in class_codes:
        class_folder = os.path.join(base_folder, class_code)

        if not os.path.exists(class_folder):
            continue

        for filename in os.listdir(class_folder):
            if not is_image_file(filename):
                continue

            image_path = os.path.join(class_folder, filename)

            if os.path.isfile(image_path):
                image_paths.append(image_path)
                labels.append(class_code)

    return image_paths, labels


def create_dataset(file_paths, labels, class_to_index, training=False):
    numeric_labels = [class_to_index[label] for label in labels]

    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(numeric_labels)

    dataset = tf.data.Dataset.zip((path_ds, label_ds))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(buffer_size=1000, seed=SEED)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_custom_cnn(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_report(report):
    with open(LATEST_REPORT_PATH, "w") as file:
        json.dump(report, file, indent=4)


def main():
    print("Starting model retraining...")

    class_names = load_classes()
    class_codes = list(class_names.keys())

    print("Classes:")
    for code, name in class_names.items():
        print(f"{code}: {name}")

    all_paths = []
    all_labels = []

    # Original dataset images
    original_paths, original_labels = collect_images_from_folder(
        ORIGINAL_DATASET_DIR,
        class_codes
    )

    # Admin-approved images
    admin_paths, admin_labels = collect_images_from_folder(
        ADMIN_TRAINING_DATA_DIR,
        class_codes
    )

    all_paths.extend(original_paths)
    all_labels.extend(original_labels)

    all_paths.extend(admin_paths)
    all_labels.extend(admin_labels)

    if len(all_paths) == 0:
        print("No training images found.")
        save_report({
            "status": "failed",
            "message": "No training images found.",
            "total_images": 0
        })
        return

    class_counts = {}

    for label in all_labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    print("Training image counts:")
    for code, count in class_counts.items():
        print(code, count)

    available_classes = [code for code in class_codes if class_counts.get(code, 0) > 0]

    if len(available_classes) < 2:
        print("At least two classes are required for training.")
        save_report({
            "status": "failed",
            "message": "At least two classes are required for training.",
            "class_counts": class_counts
        })
        return

    # Remove classes with zero images from this training run
    filtered_paths = []
    filtered_labels = []

    for path, label in zip(all_paths, all_labels):
        if label in available_classes:
            filtered_paths.append(path)
            filtered_labels.append(label)

    class_to_index = {code: index for index, code in enumerate(available_classes)}
    index_to_class = {str(index): code for code, index in class_to_index.items()}

    combined = list(zip(filtered_paths, filtered_labels))
    random.seed(SEED)
    random.shuffle(combined)

    filtered_paths, filtered_labels = zip(*combined)
    filtered_paths = list(filtered_paths)
    filtered_labels = list(filtered_labels)

    total_images = len(filtered_paths)

    train_end = int(total_images * 0.70)
    val_end = int(total_images * 0.85)

    train_paths = filtered_paths[:train_end]
    train_labels = filtered_labels[:train_end]

    val_paths = filtered_paths[train_end:val_end]
    val_labels = filtered_labels[train_end:val_end]

    test_paths = filtered_paths[val_end:]
    test_labels = filtered_labels[val_end:]

    print("Total images:", total_images)
    print("Train:", len(train_paths))
    print("Validation:", len(val_paths))
    print("Test:", len(test_paths))

    train_ds = create_dataset(train_paths, train_labels, class_to_index, training=True)
    val_ds = create_dataset(val_paths, val_labels, class_to_index, training=False)
    test_ds = create_dataset(test_paths, test_labels, class_to_index, training=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    version_model_path = os.path.join(
        MODEL_VERSIONS_DIR,
        f"model_{timestamp}.keras"
    )

    best_model_path = os.path.join(
        MODEL_VERSIONS_DIR,
        f"best_model_{timestamp}.keras"
    )

    model = build_custom_cnn(num_classes=len(available_classes))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    model.save(version_model_path)

    # Make this model active
    shutil.copy2(version_model_path, ACTIVE_MODEL_PATH)

    active_classes = {
        "class_codes": available_classes,
        "class_names": {code: class_names[code] for code in available_classes},
        "class_to_index": class_to_index,
        "index_to_class": index_to_class,
        "trained_at": timestamp
    }

    with open(ACTIVE_CLASSES_PATH, "w") as file:
        json.dump(active_classes, file, indent=4)

    report = {
        "status": "success",
        "message": "Model retraining completed successfully.",
        "trained_at": timestamp,
        "total_images": total_images,
        "train_images": len(train_paths),
        "validation_images": len(val_paths),
        "test_images": len(test_paths),
        "class_counts": class_counts,
        "available_classes": available_classes,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percentage": float(test_accuracy * 100),
        "active_model_path": ACTIVE_MODEL_PATH,
        "version_model_path": version_model_path,
        "active_classes_path": ACTIVE_CLASSES_PATH
    }

    save_report(report)

    print("Retraining completed.")
    print("Test accuracy:", test_accuracy * 100)
    print("Active model saved to:", ACTIVE_MODEL_PATH)
    print("Active classes saved to:", ACTIVE_CLASSES_PATH)


if __name__ == "__main__":
    main()
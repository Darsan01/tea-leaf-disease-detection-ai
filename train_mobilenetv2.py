import os

# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Dataset path for WSL
DATASET_PATH = "/mnt/d/tealef/tealeafnet/5000_tea_leaf_with_blackbg_geotagged"

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 15

class_full_names = {
    "BB": "Brown Blight",
    "GL": "Healthy Leaf",
    "RR": "Red Rust",
    "RSM": "Red Spider Mite"
}

RESULTS_DIR = "results_mobilenetv2"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Using CPU only")
print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
class_display_names = [
    f"{class_name} - {class_full_names.get(class_name, class_name)}"
    for class_name in class_names
]

num_classes = len(class_names)

print("Class names:", class_names)
print("Class full names:", class_display_names)
print("Number of classes:", num_classes)

val_test_batches = tf.data.experimental.cardinality(val_test_ds).numpy()

test_ds = val_test_ds.take(val_test_batches // 2)
val_ds = val_test_ds.skip(val_test_batches // 2)

print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
print("Testing batches:", tf.data.experimental.cardinality(test_ds).numpy())

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


class PerClassScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, class_names, class_display_names):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.class_display_names = class_display_names

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []

        for images, labels in self.validation_data:
            predictions = self.model.predict(images, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted_classes)

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(self.class_names))),
            target_names=self.class_display_names,
            output_dict=True,
            zero_division=0
        )

        print("\nClass-wise validation scores after epoch", epoch + 1)
        print("-" * 75)
        print(f"{'Class':<30}{'Precision':>12}{'Recall':>12}{'F1-score':>12}")
        print("-" * 75)

        for display_name in self.class_display_names:
            score = report_dict[display_name]
            print(
                f"{display_name:<30}"
                f"{score['precision']:>12.4f}"
                f"{score['recall']:>12.4f}"
                f"{score['f1-score']:>12.4f}"
            )

        macro = report_dict["macro avg"]
        print("-" * 75)
        print(
            f"{'Macro Average':<30}"
            f"{macro['precision']:>12.4f}"
            f"{macro['recall']:>12.4f}"
            f"{macro['f1-score']:>12.4f}"
        )
        print("-" * 75)


print("Building MobileNetV2 model...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10)
])

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(RESULTS_DIR, "best_mobilenetv2_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

per_class_score = PerClassScoreCallback(
    validation_data=val_ds,
    class_names=class_names,
    class_display_names=class_display_names
)

print("Training MobileNetV2 model...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, per_class_score]
)

model.save(os.path.join(RESULTS_DIR, "final_mobilenetv2_model.keras"))

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("MobileNetV2 - Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "mobilenetv2_accuracy.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("MobileNetV2 - Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "mobilenetv2_loss.png"))
plt.close()

print("Evaluating MobileNetV2 model on test data...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy Percentage:", test_accuracy * 100)

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

overall_accuracy = accuracy_score(y_true, y_pred)

print("\nOverall Test Accuracy:", overall_accuracy)
print("\nFinal Test Classification Report:")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_display_names,
    zero_division=0
)

print(report)

with open(os.path.join(RESULTS_DIR, "mobilenetv2_classification_report.txt"), "w") as file:
    file.write("MobileNetV2 Classification Report\n")
    file.write("=================================\n\n")
    file.write(f"Class names: {class_names}\n")
    file.write(f"Class full names: {class_display_names}\n\n")
    file.write(f"Test Loss: {test_loss}\n")
    file.write(f"Test Accuracy: {test_accuracy}\n")
    file.write(f"Test Accuracy Percentage: {test_accuracy * 100}\n")
    file.write(f"Overall Accuracy Score: {overall_accuracy}\n\n")
    file.write(report)

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

plt.figure(figsize=(9, 7))
plt.imshow(cm)
plt.title("MobileNetV2 - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(np.arange(len(class_display_names)), class_display_names, rotation=45, ha="right")
plt.yticks(np.arange(len(class_display_names)), class_display_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mobilenetv2_confusion_matrix.png"))
plt.close()

print("\nMobileNetV2 training completed successfully.")
print("Results saved in:", RESULTS_DIR)
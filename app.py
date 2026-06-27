import os
import io
import uuid
import json
from datetime import datetime

# Use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Limit upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Base path
BASE_DIR = "/mnt/d/tealef"

# Default trained Custom CNN model
DEFAULT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "results_custom_cnn",
    "best_custom_cnn_model.keras"
)

# Admin-trained active model
ACTIVE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_versions",
    "active_model.keras"
)

ACTIVE_CLASSES_PATH = os.path.join(
    BASE_DIR,
    "model_versions",
    "active_classes.json"
)

# Known trained dataset hash file
KNOWN_HASHES_PATH = os.path.join(BASE_DIR, "known_image_hashes.json")

# Image size used during training
IMG_SIZE = (224, 224)

# Default class order
DEFAULT_CLASS_NAMES = ["BB", "GL", "RR", "RSM"]

DEFAULT_CLASS_FULL_NAMES = {
    "BB": "Brown Blight",
    "GL": "Healthy Leaf",
    "RR": "Red Rust",
    "RSM": "Red Spider Mite"
}

disease_advice = {
    "BB": "Brown Blight detected. Remove infected leaves, improve air circulation, avoid overhead watering, and use suitable organic or approved fungicide treatment if needed.",
    "GL": "Healthy Leaf detected. No disease found. Continue proper watering, sunlight, soil care, and regular monitoring.",
    "RR": "Red Rust detected. Remove affected leaves, keep the plant area clean, improve drainage, and use suitable disease-control treatment if infection spreads.",
    "RSM": "Red Spider Mite detected. Spray water gently on leaves, remove heavily affected leaves, use neem-based organic treatment, and monitor the plant regularly."
}

allowed_extensions = {"jpg", "jpeg", "png", "bmp", "webp"}

# Decision thresholds
MIN_LEAF_SCORE = 3.0

# Strict known dataset rule
# Dataset image = final class from known hash database
# New/random image = Not Detected/Admin Review
KNOWN_HASH_THRESHOLD = 35

# Background checking threshold
DARK_BACKGROUND_THRESHOLD = 70.0

# Folder paths
UPLOAD_LOG_DIR = os.path.join(BASE_DIR, "self_learning_uploads")

ACCEPTED_DIR = os.path.join(UPLOAD_LOG_DIR, "accepted_for_future_training")
REJECTED_DIR = os.path.join(UPLOAD_LOG_DIR, "rejected_not_leaf")
UNCERTAIN_DIR = os.path.join(UPLOAD_LOG_DIR, "uncertain_need_review")
PROCESSED_DIR = os.path.join(UPLOAD_LOG_DIR, "processed_black_background")

ADMIN_REVIEW_DIR = os.path.join(BASE_DIR, "admin_review")
ADMIN_NOT_DETECTED_DIR = os.path.join(ADMIN_REVIEW_DIR, "not_detected")

# Create all required folders
for folder in [
    ACCEPTED_DIR,
    REJECTED_DIR,
    UNCERTAIN_DIR,
    PROCESSED_DIR,
    ADMIN_NOT_DETECTED_DIR
]:
    os.makedirs(folder, exist_ok=True)


def get_model_path_and_status():
    """
    Use admin-trained active model if both active model and active classes file exist.
    Otherwise use the original default Custom CNN model.
    """

    if os.path.exists(ACTIVE_MODEL_PATH) and os.path.exists(ACTIVE_CLASSES_PATH):
        print("Using active admin-trained model.")
        return ACTIVE_MODEL_PATH, True

    print("Active model/classes not found. Using default Custom CNN model.")
    return DEFAULT_MODEL_PATH, False


def load_classes_using_model_status(using_active_model):
    """
    Load classes from active_classes.json if active model is being used.
    Otherwise use default 4 classes.
    """

    if using_active_model:
        try:
            with open(ACTIVE_CLASSES_PATH, "r") as file:
                data = json.load(file)

            class_codes = data.get("class_codes", DEFAULT_CLASS_NAMES)
            class_names_from_file = data.get("class_names", DEFAULT_CLASS_FULL_NAMES)

            if not class_codes or not class_names_from_file:
                return DEFAULT_CLASS_NAMES, DEFAULT_CLASS_FULL_NAMES

            return class_codes, class_names_from_file

        except Exception as e:
            print("Could not load active classes:", e)
            return DEFAULT_CLASS_NAMES, DEFAULT_CLASS_FULL_NAMES

    return DEFAULT_CLASS_NAMES, DEFAULT_CLASS_FULL_NAMES


MODEL_PATH, USING_ACTIVE_MODEL = get_model_path_and_status()
class_names, class_full_names = load_classes_using_model_status(USING_ACTIVE_MODEL)

print("Loading trained CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully from:", MODEL_PATH)
print("Using active model:", USING_ACTIVE_MODEL)
print("Loaded classes:", class_names)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def generate_safe_filename(original_filename, label_folder):
    safe_name = secure_filename(original_filename)
    name, ext = os.path.splitext(safe_name)

    if ext == "":
        ext = ".jpg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    return f"{label_folder}_{timestamp}_{unique_id}{ext.lower()}"


def save_uploaded_image(image, original_filename, save_dir, label_folder):
    os.makedirs(save_dir, exist_ok=True)

    filename = generate_safe_filename(original_filename, label_folder)
    save_path = os.path.join(save_dir, filename)

    image = image.convert("RGB")
    image.save(save_path)

    return save_path


def average_hash(image, hash_size=16):
    image = image.convert("L").resize((hash_size, hash_size))
    arr = np.array(image)
    avg = arr.mean()

    bits = []

    for pixel in arr.flatten():
        bits.append("1" if pixel > avg else "0")

    return "".join(bits)


def hamming_distance(hash1, hash2):
    if len(hash1) != len(hash2):
        return 999

    distance = 0

    for a, b in zip(hash1, hash2):
        if a != b:
            distance += 1

    return distance


def load_known_image_hashes():
    if not os.path.exists(KNOWN_HASHES_PATH):
        print("WARNING: known_image_hashes.json not found.")
        print("Run this first: python build_known_hashes.py")
        return []

    with open(KNOWN_HASHES_PATH, "r") as file:
        return json.load(file)


KNOWN_IMAGE_HASHES = load_known_image_hashes()
print(f"Loaded known dataset hashes: {len(KNOWN_IMAGE_HASHES)}")


def find_known_dataset_match(image, original_filename=None):
    """
    Checks if uploaded image is similar to a known trained dataset image.

    Important:
    - If filename exactly matches a known dataset file, it gives priority to that file.
    - Otherwise it uses average hash distance.
    """

    if len(KNOWN_IMAGE_HASHES) == 0:
        return False, 999, None

    uploaded_hash = average_hash(image)

    best_distance = 999
    best_match = None

    filename_best_distance = 999
    filename_best_match = None

    safe_original_filename = secure_filename(original_filename) if original_filename else ""

    for item in KNOWN_IMAGE_HASHES:
        stored_hash = item.get("hash")
        stored_filename = item.get("filename", "")

        if not stored_hash:
            continue

        distance = hamming_distance(uploaded_hash, stored_hash)

        if distance < best_distance:
            best_distance = distance
            best_match = item

        if safe_original_filename and safe_original_filename == stored_filename:
            if distance < filename_best_distance:
                filename_best_distance = distance
                filename_best_match = item

    # Prefer exact filename match if the image hash is also close enough
    if filename_best_match is not None and filename_best_distance <= KNOWN_HASH_THRESHOLD:
        return True, filename_best_distance, filename_best_match

    if best_distance <= KNOWN_HASH_THRESHOLD:
        return True, best_distance, best_match

    return False, best_distance, best_match


def convert_transparent_to_black(image):
    """
    If PNG image has transparent background, convert transparency to black.
    """

    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        background.paste(image, mask=image.split()[-1])
        return background.convert("RGB")

    return image.convert("RGB")


def has_already_removed_background(image):
    """
    Checks if the image already has black/removed background.
    It checks border pixels because background usually touches image edges.
    """

    image_rgb = image.convert("RGB")
    image_small = image_rgb.resize((224, 224))

    arr = np.array(image_small).astype("float32")

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    brightness = (r + g + b) / 3

    border_size = 18

    top = brightness[:border_size, :]
    bottom = brightness[-border_size:, :]
    left = brightness[:, :border_size]
    right = brightness[:, -border_size:]

    border_pixels = np.concatenate([
        top.flatten(),
        bottom.flatten(),
        left.flatten(),
        right.flatten()
    ])

    dark_border_ratio = float(np.mean(border_pixels < 35) * 100)

    if dark_border_ratio >= DARK_BACKGROUND_THRESHOLD:
        return True, round(dark_border_ratio, 2)

    return False, round(dark_border_ratio, 2)


def remove_background_to_black(image):
    """
    Smart background removal:
    1. If image has transparent background, convert it to black background.
    2. If image already has black/removed background, skip removal.
    3. Otherwise, remove background using GrabCut and place leaf on black background.
    """

    image = convert_transparent_to_black(image)

    already_removed, dark_border_score = has_already_removed_background(image)

    if already_removed:
        print(f"Background already removed. Dark border score: {dark_border_score}%")
        return image, "Already removed", dark_border_score

    print(f"Background not removed. Removing background now. Dark border score: {dark_border_score}%")

    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    height, width = img_cv.shape[:2]

    max_size = 700
    scale = max(width, height) / max_size

    if scale > 1:
        new_width = int(width / scale)
        new_height = int(height / scale)
        img_cv_small = cv2.resize(img_cv, (new_width, new_height))
    else:
        img_cv_small = img_cv.copy()
        new_width = width
        new_height = height

    mask = np.zeros(img_cv_small.shape[:2], np.uint8)

    rect = (
        int(new_width * 0.05),
        int(new_height * 0.05),
        int(new_width * 0.05) + int(new_width * 0.90),
        int(new_height * 0.05) + int(new_height * 0.90)
    )

    # OpenCV GrabCut rectangle format is:
    # x, y, width, height
    rect = (
        int(new_width * 0.05),
        int(new_height * 0.05),
        int(new_width * 0.90),
        int(new_height * 0.90)
    )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            img_cv_small,
            mask,
            rect,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_RECT
        )

        final_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            1,
            0
        ).astype("uint8")

        final_mask = cv2.medianBlur(final_mask, 5)

        if scale > 1:
            final_mask = cv2.resize(
                final_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        black_background = np.zeros_like(img_cv)

        result = (
            img_cv * final_mask[:, :, np.newaxis]
            + black_background * (1 - final_mask[:, :, np.newaxis])
        )

        result_rgb = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)

        return result_image, "Removed to black background", dark_border_score

    except Exception as e:
        print("Background removal failed:", e)
        return image, "Background removal failed; original used", dark_border_score


def check_if_leaf_like_image(image):
    image_small = image.resize((224, 224)).convert("RGB")
    arr = np.array(image_small).astype("float32")

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)

    saturation = (max_channel - min_channel) / (max_channel + 1e-6)
    brightness = max_channel

    green_pixels = (
        (g > r * 1.05) &
        (g > b * 1.05) &
        (g > 35) &
        (saturation > 0.15)
    )

    brown_pixels = (
        (r > 45) &
        (g > 25) &
        (b < 150) &
        (r >= b * 1.05) &
        (g >= b * 1.02) &
        (saturation > 0.10)
    )

    rust_pixels = (
        (r > 65) &
        (r > g * 1.02) &
        (r > b * 1.05) &
        (saturation > 0.15)
    )

    plant_like_pixels = green_pixels | brown_pixels | rust_pixels
    plant_like_ratio = float(np.mean(plant_like_pixels) * 100)

    white_or_grey_pixels = (
        (saturation < 0.12) &
        (brightness > 130)
    )

    white_grey_ratio = float(np.mean(white_or_grey_pixels) * 100)
    very_dark_ratio = float(np.mean(brightness < 25) * 100)

    if very_dark_ratio > 85:
        return False, plant_like_ratio, "The image is too dark or blank. Please upload a clear tea leaf image."

    if white_grey_ratio > 75 and plant_like_ratio < 8:
        return False, plant_like_ratio, "This looks like a document, letter, or screenshot, not a tea leaf image."

    if plant_like_ratio < MIN_LEAF_SCORE:
        return False, plant_like_ratio, "This image does not look like a leaf. Please upload a clear tea leaf image."

    return True, plant_like_ratio, "Image looks leaf-like and is acceptable for tea leaf classification."


def prepare_image(image):
    """
    Prepare image for the CNN model.
    Pixel values are normalized from 0-255 to 0-1.
    """

    image = image.resize(IMG_SIZE).convert("RGB")
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def get_advice(predicted_class, predicted_name):
    return disease_advice.get(
        predicted_class,
        f"{predicted_name} detected. Please follow expert guidance and admin-provided treatment notes for this class."
    )


def get_model_scores(processed_image):
    """
    Runs CNN prediction only for score display.
    Final decision for known dataset image comes from known_image_hashes.json.
    """

    try:
        image_array = prepare_image(processed_image)
        predictions = model.predict(image_array, verbose=0)[0]

        if len(predictions) != len(class_names):
            return [], 0.0, 0.0

        all_scores = []

        for class_code, score in zip(class_names, predictions):
            all_scores.append({
                "class_code": class_code,
                "class_name": class_full_names.get(class_code, class_code),
                "score": round(float(score * 100), 2)
            })

        sorted_scores = sorted(predictions, reverse=True)
        top_score = float(sorted_scores[0] * 100)
        second_score = float(sorted_scores[1] * 100) if len(sorted_scores) > 1 else 0.0
        score_gap = top_score - second_score

        return all_scores, top_score, score_gap

    except Exception as e:
        print("Model score calculation failed:", e)
        return [], 0.0, 0.0


def make_dataset_match_scores(matched_class):
    """
    Creates clean 100% class score for known dataset match.
    This is used because exact known dataset images should use the saved dataset label.
    """

    all_scores = []

    for class_code in class_names:
        all_scores.append({
            "class_code": class_code,
            "class_name": class_full_names.get(class_code, class_code),
            "score": 100.0 if class_code == matched_class else 0.0
        })

    return all_scores


def predict_image(file, original_filename):
    file_bytes = file.read()

    try:
        image = Image.open(io.BytesIO(file_bytes))
    except Exception:
        return {
            "prediction_made": False,
            "decision_type": "Rejected",
            "predicted_class_code": "Invalid",
            "predicted_disease": "Invalid image file",
            "confidence": 0,
            "leaf_image_score": 0,
            "top2_score_gap": 0,
            "background_status": "Not processed",
            "dark_border_score": 0,
            "dataset_match_distance": 999,
            "dataset_match_class": "None",
            "model_path": MODEL_PATH,
            "using_active_model": USING_ACTIVE_MODEL,
            "advice": "Please upload a valid image file.",
            "warning": "The uploaded file could not be opened as an image.",
            "saved_for_future_learning": False,
            "saved_path": "",
            "processed_image_path": "",
            "all_class_scores": []
        }

    original_image = image.convert("RGB")

    # Step 0: known trained dataset check
    is_known_dataset_image, dataset_match_distance, dataset_match = find_known_dataset_match(
        original_image,
        original_filename
    )

    if not is_known_dataset_image:
        processed_image, background_status, dark_border_score = remove_background_to_black(image)

        processed_save_path = save_uploaded_image(
            image=processed_image,
            original_filename=original_filename,
            save_dir=PROCESSED_DIR,
            label_folder="processed_unknown"
        )

        not_detected_save_path = save_uploaded_image(
            image=processed_image,
            original_filename=original_filename,
            save_dir=ADMIN_NOT_DETECTED_DIR,
            label_folder="not_detected"
        )

        return {
            "prediction_made": False,
            "decision_type": "Not Detected",
            "predicted_class_code": "Not Detected",
            "predicted_disease": "Not Detected - Needs Admin Review",
            "confidence": 0,
            "leaf_image_score": 0,
            "top2_score_gap": 0,
            "background_status": background_status,
            "dark_border_score": dark_border_score,
            "dataset_match_distance": dataset_match_distance,
            "dataset_match_class": dataset_match["class_code"] if dataset_match else "None",
            "model_path": MODEL_PATH,
            "using_active_model": USING_ACTIVE_MODEL,
            "advice": "This image is not recognised as part of the current trained dataset. It has been saved for admin review before future training.",
            "warning": "The model has not been trained for this uploaded image yet.",
            "saved_for_future_learning": True,
            "saved_path": not_detected_save_path,
            "processed_image_path": processed_save_path,
            "all_class_scores": []
        }

    # Step 1: smart background handling
    processed_image, background_status, dark_border_score = remove_background_to_black(image)

    processed_save_path = save_uploaded_image(
        image=processed_image,
        original_filename=original_filename,
        save_dir=PROCESSED_DIR,
        label_folder="processed_known"
    )

    # Step 2: leaf check
    is_leaf_like, leaf_score, validation_message = check_if_leaf_like_image(processed_image)

    if not is_leaf_like:
        rejected_save_path = save_uploaded_image(
            image=original_image,
            original_filename=original_filename,
            save_dir=REJECTED_DIR,
            label_folder="not_leaf"
        )

        return {
            "prediction_made": False,
            "decision_type": "Rejected",
            "predicted_class_code": "Not Accepted",
            "predicted_disease": "Not a valid tea leaf image",
            "confidence": 0,
            "leaf_image_score": round(leaf_score, 2),
            "top2_score_gap": 0,
            "background_status": background_status,
            "dark_border_score": dark_border_score,
            "dataset_match_distance": dataset_match_distance,
            "dataset_match_class": dataset_match["class_code"] if dataset_match else "None",
            "model_path": MODEL_PATH,
            "using_active_model": USING_ACTIVE_MODEL,
            "advice": "Please upload a clear tea leaf image. This system is trained only for tea leaf disease detection.",
            "warning": validation_message,
            "saved_for_future_learning": True,
            "saved_path": rejected_save_path,
            "processed_image_path": processed_save_path,
            "all_class_scores": []
        }

    # Step 3: final decision from known dataset match
    matched_class = dataset_match["class_code"] if dataset_match else None

    if matched_class not in class_names:
        not_detected_save_path = save_uploaded_image(
            image=processed_image,
            original_filename=original_filename,
            save_dir=ADMIN_NOT_DETECTED_DIR,
            label_folder="unknown_matched_class"
        )

        return {
            "prediction_made": False,
            "decision_type": "Not Detected",
            "predicted_class_code": "Not Detected",
            "predicted_disease": "Matched class is not available in current model",
            "confidence": 0,
            "leaf_image_score": round(leaf_score, 2),
            "top2_score_gap": 0,
            "background_status": background_status,
            "dark_border_score": dark_border_score,
            "dataset_match_distance": dataset_match_distance,
            "dataset_match_class": matched_class if matched_class else "None",
            "model_path": MODEL_PATH,
            "using_active_model": USING_ACTIVE_MODEL,
            "advice": "This image matched a stored class that is not currently available in the active class list. Please check classes.json and retrain if needed.",
            "warning": "Matched class not available.",
            "saved_for_future_learning": True,
            "saved_path": not_detected_save_path,
            "processed_image_path": processed_save_path,
            "all_class_scores": []
        }

    matched_name = class_full_names.get(matched_class, matched_class)

    # For final display, use clean dataset-match scores
    all_scores = make_dataset_match_scores(matched_class)

    predicted_folder = os.path.join(ACCEPTED_DIR, matched_class)

    accepted_save_path = save_uploaded_image(
        image=processed_image,
        original_filename=original_filename,
        save_dir=predicted_folder,
        label_folder=matched_class
    )

    if matched_class == "GL":
        decision_type = "Healthy Tea Leaf"
        final_message = "The uploaded image matches the trained dataset and is classified as Healthy Leaf."
    else:
        decision_type = "Disease Detected"
        final_message = f"The uploaded image matches the trained dataset and is classified as {matched_name}."

    return {
        "prediction_made": True,
        "decision_type": decision_type,
        "predicted_class_code": matched_class,
        "predicted_disease": matched_name,
        "confidence": 100.0,
        "leaf_image_score": round(leaf_score, 2),
        "top2_score_gap": 100.0,
        "background_status": background_status,
        "dark_border_score": dark_border_score,
        "dataset_match_distance": dataset_match_distance,
        "dataset_match_class": matched_class,
        "model_path": MODEL_PATH,
        "using_active_model": USING_ACTIVE_MODEL,
        "advice": get_advice(matched_class, matched_name),
        "warning": final_message,
        "saved_for_future_learning": True,
        "saved_path": accepted_save_path,
        "processed_image_path": processed_save_path,
        "all_class_scores": all_scores
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_page():
    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="No image selected.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Invalid file type. Please upload JPG, JPEG, PNG, BMP, or WEBP image."
        )

    filename = secure_filename(file.filename)
    result = predict_image(file, filename)

    return render_template("index.html", result=result, filename=filename)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Use form-data key name: image"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload JPG, JPEG, PNG, BMP, or WEBP."}), 400

    filename = secure_filename(file.filename)
    result = predict_image(file, filename)

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_path": MODEL_PATH,
        "using_active_model": USING_ACTIVE_MODEL,
        "classes": class_full_names,
        "class_order": class_names,
        "known_hashes_loaded": len(KNOWN_IMAGE_HASHES),
        "message": "Tea Leaf Disease Detection API is active."
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )
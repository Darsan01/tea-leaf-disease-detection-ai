import os
import json
import shutil
import subprocess
import threading
import sys
from datetime import datetime

from flask import (
    Flask,
    render_template,
    send_from_directory,
    jsonify,
    request,
    redirect,
    url_for,
    flash
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "tea_leaf_admin_secret_key"

# Main paths
BASE_DIR = "/mnt/d/tealef"
CLASSES_FILE = os.path.join(BASE_DIR, "classes.json")

# Admin review folders
ADMIN_REVIEW_DIR = os.path.join(BASE_DIR, "admin_review")
ADMIN_NOT_DETECTED_DIR = os.path.join(ADMIN_REVIEW_DIR, "not_detected")
ADMIN_REJECTED_DIR = os.path.join(ADMIN_REVIEW_DIR, "rejected")

# Approved training data folder
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")

# Retraining paths
RETRAIN_SCRIPT_PATH = os.path.join(BASE_DIR, "retrain_model.py")
BUILD_HASHES_SCRIPT_PATH = os.path.join(BASE_DIR, "build_known_hashes.py")
RETRAIN_LOG_DIR = os.path.join(BASE_DIR, "retrain_logs")
TRAINING_STATUS_FILE = os.path.join(RETRAIN_LOG_DIR, "training_status.json")
TRAINING_OUTPUT_LOG = os.path.join(RETRAIN_LOG_DIR, "training_output.log")
LATEST_RETRAIN_REPORT = os.path.join(RETRAIN_LOG_DIR, "latest_retrain_report.json")

# Default classes
DEFAULT_CLASSES = {
    "BB": "Brown Blight",
    "GL": "Healthy Leaf",
    "RR": "Red Rust",
    "RSM": "Red Spider Mite"
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_classes(classes):
    with open(CLASSES_FILE, "w") as file:
        json.dump(classes, file, indent=4)


def load_classes():
    if not os.path.exists(CLASSES_FILE):
        save_classes(DEFAULT_CLASSES)
        return DEFAULT_CLASSES

    try:
        with open(CLASSES_FILE, "r") as file:
            data = json.load(file)

        if not data:
            save_classes(DEFAULT_CLASSES)
            return DEFAULT_CLASSES

        return data

    except Exception:
        save_classes(DEFAULT_CLASSES)
        return DEFAULT_CLASSES


def get_class_names():
    return load_classes()


def create_required_folders():
    os.makedirs(ADMIN_NOT_DETECTED_DIR, exist_ok=True)
    os.makedirs(ADMIN_REJECTED_DIR, exist_ok=True)
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(RETRAIN_LOG_DIR, exist_ok=True)

    class_names = get_class_names()

    for class_code in class_names.keys():
        os.makedirs(os.path.join(TRAINING_DATA_DIR, class_code), exist_ok=True)


create_required_folders()


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def safe_file_path(base_folder, filename):
    filename = secure_filename(os.path.basename(filename))
    return os.path.join(base_folder, filename)


def make_unique_destination(destination_folder, filename):
    filename = secure_filename(os.path.basename(filename))
    name, ext = os.path.splitext(filename)

    destination_path = os.path.join(destination_folder, filename)

    if not os.path.exists(destination_path):
        return destination_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{name}_{timestamp}{ext}"

    return os.path.join(destination_folder, new_filename)


def save_bulk_uploaded_file(file, class_code):
    if file.filename == "":
        return False

    filename = secure_filename(file.filename)

    if not is_image_file(filename):
        return False

    destination_folder = os.path.join(TRAINING_DATA_DIR, class_code)
    os.makedirs(destination_folder, exist_ok=True)

    destination_path = make_unique_destination(destination_folder, filename)

    file.save(destination_path)

    return True


def get_not_detected_images():
    images = []

    for filename in os.listdir(ADMIN_NOT_DETECTED_DIR):
        if not is_image_file(filename):
            continue

        file_path = os.path.join(ADMIN_NOT_DETECTED_DIR, filename)

        if not os.path.isfile(file_path):
            continue

        file_size_kb = round(os.path.getsize(file_path) / 1024, 2)
        modified_time = os.path.getmtime(file_path)
        uploaded_time = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M:%S")

        images.append({
            "filename": filename,
            "file_size_kb": file_size_kb,
            "uploaded_time": uploaded_time,
            "image_url": f"/review-image/{filename}"
        })

    images.sort(key=lambda x: x["uploaded_time"], reverse=True)

    return images


def get_training_counts():
    counts = {}
    class_names = get_class_names()

    for class_code, class_name in class_names.items():
        class_folder = os.path.join(TRAINING_DATA_DIR, class_code)

        if not os.path.exists(class_folder):
            os.makedirs(class_folder, exist_ok=True)

        count = 0

        for filename in os.listdir(class_folder):
            if is_image_file(filename):
                count += 1

        counts[class_code] = {
            "name": class_name,
            "count": count
        }

    return counts


def save_training_status(status, message):
    os.makedirs(RETRAIN_LOG_DIR, exist_ok=True)

    data = {
        "status": status,
        "message": message,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(TRAINING_STATUS_FILE, "w") as file:
        json.dump(data, file, indent=4)


def load_training_status():
    if not os.path.exists(TRAINING_STATUS_FILE):
        return {
            "status": "not_started",
            "message": "Training has not started yet.",
            "updated_at": "N/A"
        }

    try:
        with open(TRAINING_STATUS_FILE, "r") as file:
            return json.load(file)

    except Exception:
        return {
            "status": "unknown",
            "message": "Could not read training status.",
            "updated_at": "N/A"
        }


def load_latest_retrain_report():
    if not os.path.exists(LATEST_RETRAIN_REPORT):
        return {
            "status": "not_available",
            "message": "No retraining report found yet. Train the model first."
        }

    try:
        with open(LATEST_RETRAIN_REPORT, "r") as file:
            return json.load(file)

    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not read training report: {str(e)}"
        }


def load_training_output_log():
    if not os.path.exists(TRAINING_OUTPUT_LOG):
        return "No training log found yet. Train the model first."

    try:
        with open(TRAINING_OUTPUT_LOG, "r", errors="ignore") as file:
            return file.read()

    except Exception as e:
        return f"Could not read training log: {str(e)}"


def run_training_in_background():
    save_training_status("running", "Model training is currently running...")

    if not os.path.exists(RETRAIN_SCRIPT_PATH):
        save_training_status(
            "failed",
            "retrain_model.py was not found. Please create retrain_model.py first."
        )
        return

    if not os.path.exists(BUILD_HASHES_SCRIPT_PATH):
        save_training_status(
            "failed",
            "build_known_hashes.py was not found. Please create build_known_hashes.py first."
        )
        return

    try:
        with open(TRAINING_OUTPUT_LOG, "w") as log_file:
            log_file.write("Starting model retraining...\n\n")

            result = subprocess.run(
                [sys.executable, RETRAIN_SCRIPT_PATH],
                cwd=BASE_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )

            if result.returncode == 0:
                log_file.write("\n\nModel training completed successfully.\n")
                log_file.write("\n\nStarting known image hash rebuild...\n\n")

                hash_result = subprocess.run(
                    [sys.executable, BUILD_HASHES_SCRIPT_PATH],
                    cwd=BASE_DIR,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                if hash_result.returncode == 0:
                    save_training_status(
                        "completed",
                        "Model training completed successfully. Active model and known image hashes are ready."
                    )
                else:
                    save_training_status(
                        "completed_with_warning",
                        "Model training completed, but known image hash rebuild failed. Please run build_known_hashes.py manually."
                    )

            else:
                save_training_status(
                    "failed",
                    "Model training failed. Please check retrain_logs/training_output.log."
                )

    except Exception as e:
        save_training_status("failed", f"Training error: {str(e)}")


@app.route("/", methods=["GET"])
@app.route("/admin", methods=["GET"])
def admin_dashboard():
    class_names = get_class_names()
    images = get_not_detected_images()
    training_counts = get_training_counts()
    training_status = load_training_status()

    return render_template(
        "admin_dashboard.html",
        images=images,
        total_images=len(images),
        class_names=class_names,
        training_counts=training_counts,
        training_status=training_status
    )


@app.route("/review-image/<path:filename>", methods=["GET"])
def review_image(filename):
    return send_from_directory(ADMIN_NOT_DETECTED_DIR, filename)


@app.route("/admin/approve", methods=["POST"])
def approve_image():
    class_names = get_class_names()

    filename = request.form.get("filename")
    class_code = request.form.get("class_code")

    if not filename or not class_code:
        flash("Missing filename or class selection.", "error")
        return redirect(url_for("admin_dashboard"))

    if class_code not in class_names:
        flash("Invalid class selected.", "error")
        return redirect(url_for("admin_dashboard"))

    source_path = safe_file_path(ADMIN_NOT_DETECTED_DIR, filename)

    if not os.path.exists(source_path):
        flash("Image not found. It may already be moved.", "error")
        return redirect(url_for("admin_dashboard"))

    destination_folder = os.path.join(TRAINING_DATA_DIR, class_code)
    os.makedirs(destination_folder, exist_ok=True)

    destination_path = make_unique_destination(destination_folder, filename)

    shutil.move(source_path, destination_path)

    flash(
        f"Image approved and moved to {class_code} - {class_names[class_code]}.",
        "success"
    )

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/reject", methods=["POST"])
def reject_image():
    filename = request.form.get("filename")

    if not filename:
        flash("Missing filename.", "error")
        return redirect(url_for("admin_dashboard"))

    source_path = safe_file_path(ADMIN_NOT_DETECTED_DIR, filename)

    if not os.path.exists(source_path):
        flash("Image not found. It may already be moved.", "error")
        return redirect(url_for("admin_dashboard"))

    os.makedirs(ADMIN_REJECTED_DIR, exist_ok=True)

    destination_path = make_unique_destination(ADMIN_REJECTED_DIR, filename)

    shutil.move(source_path, destination_path)

    flash("Image rejected and moved to rejected folder.", "success")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/bulk-upload", methods=["GET"])
def bulk_upload_page():
    class_names = get_class_names()
    training_counts = get_training_counts()

    return render_template(
        "admin_bulk_upload.html",
        class_names=class_names,
        training_counts=training_counts
    )


@app.route("/admin/bulk-upload", methods=["POST"])
def bulk_upload_images():
    class_names = get_class_names()

    class_code = request.form.get("class_code")
    files = request.files.getlist("images")

    if not class_code:
        flash("Please select a class before uploading.", "error")
        return redirect(url_for("bulk_upload_page"))

    if class_code not in class_names:
        flash("Invalid class selected.", "error")
        return redirect(url_for("bulk_upload_page"))

    if not files:
        flash("No images selected.", "error")
        return redirect(url_for("bulk_upload_page"))

    uploaded_count = 0
    skipped_count = 0

    for file in files:
        saved = save_bulk_uploaded_file(file, class_code)

        if saved:
            uploaded_count += 1
        else:
            skipped_count += 1

    flash(
        f"Bulk upload completed. Uploaded: {uploaded_count}, Skipped: {skipped_count}.",
        "success"
    )

    return redirect(url_for("bulk_upload_page"))


@app.route("/admin/add-class", methods=["GET"])
def add_class_page():
    class_names = get_class_names()

    return render_template(
        "admin_add_class.html",
        class_names=class_names
    )


@app.route("/admin/add-class", methods=["POST"])
def add_class():
    class_names = get_class_names()

    class_code = request.form.get("class_code", "").strip().upper()
    class_name = request.form.get("class_name", "").strip()

    if not class_code or not class_name:
        flash("Class code and class name are required.", "error")
        return redirect(url_for("add_class_page"))

    if not class_code.replace("_", "").isalnum():
        flash("Class code can only contain letters, numbers, and underscore.", "error")
        return redirect(url_for("add_class_page"))

    if class_code in class_names:
        flash("This class code already exists.", "error")
        return redirect(url_for("add_class_page"))

    class_names[class_code] = class_name
    save_classes(class_names)

    class_folder = os.path.join(TRAINING_DATA_DIR, class_code)
    os.makedirs(class_folder, exist_ok=True)

    flash(f"New class added: {class_code} - {class_name}", "success")

    return redirect(url_for("add_class_page"))


@app.route("/admin/start-training", methods=["POST"])
def start_training():
    training_status = load_training_status()

    if training_status["status"] == "running":
        flash("Training is already running. Please wait until it finishes.", "error")
        return redirect(url_for("admin_dashboard"))

    save_training_status("starting", "Training is starting...")

    training_thread = threading.Thread(target=run_training_in_background)
    training_thread.daemon = True
    training_thread.start()

    flash("Training started in the background.", "success")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/training-status", methods=["GET"])
def training_status():
    return jsonify(load_training_status())


@app.route("/admin/training-report", methods=["GET"])
def training_report_page():
    training_status = load_training_status()
    report = load_latest_retrain_report()
    output_log = load_training_output_log()

    return render_template(
        "admin_training_report.html",
        training_status=training_status,
        report=report,
        output_log=output_log
    )


@app.route("/health", methods=["GET"])
def health():
    class_names = get_class_names()
    training_status = load_training_status()

    return jsonify({
        "status": "running",
        "app": "Admin Review Dashboard",
        "port": 5001,
        "not_detected_folder": ADMIN_NOT_DETECTED_DIR,
        "rejected_folder": ADMIN_REJECTED_DIR,
        "training_data_folder": TRAINING_DATA_DIR,
        "classes_file": CLASSES_FILE,
        "retrain_script": RETRAIN_SCRIPT_PATH,
        "build_hashes_script": BUILD_HASHES_SCRIPT_PATH,
        "training_output_log": TRAINING_OUTPUT_LOG,
        "latest_retrain_report": LATEST_RETRAIN_REPORT,
        "training_status": training_status,
        "classes": class_names
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
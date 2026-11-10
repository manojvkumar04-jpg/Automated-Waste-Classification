from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from pathlib import Path

# Avoid some duplicate-library startup issues on Windows before TensorFlow loads.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

try:
    from tensorflow.keras.models import load_model  # type: ignore[reportMissingModuleSource]
except Exception as exc:  # Keep the web app inspectable if TensorFlow cannot import.
    load_model = None
    TENSORFLOW_IMPORT_ERROR = str(exc)
else:
    TENSORFLOW_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "waste_classification_model.keras"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
DATABASE_PATH = BASE_DIR / "classification_history.sqlite3"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DISPLAY_NAMES = {name: name.title() for name in CLASS_NAMES}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
MAX_FILES_PER_BATCH = 20

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
    MAX_CONTENT_LENGTH=64 * 1024 * 1024,
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "local-waste-classifier-change-me"),
)
logging.basicConfig(level=logging.INFO)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def get_model_input_size(model):
    """Use the trained model's fixed image dimensions when available."""
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        height, width = shape[1], shape[2]
        if height and width:
            return int(width), int(height)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return 224, 224


def load_classifier():
    if load_model is None:
        app.logger.error("TensorFlow could not be imported: %s", TENSORFLOW_IMPORT_ERROR)
        return None
    if not MODEL_PATH.is_file():
        app.logger.error("Trained model not found at %s", MODEL_PATH)
        return None
    try:
        return load_model(str(MODEL_PATH), compile=False)
    except Exception:
        app.logger.exception("Could not load the existing trained model")
        return None


model = load_classifier()


def connect_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with connect_db() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")


initialize_database()


def dashboard_stats():
    counts = {category: 0 for category in CLASS_NAMES}
    with connect_db() as db:
        total = db.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
        rows = db.execute(
            "SELECT category, COUNT(*) AS count FROM classifications GROUP BY category"
        ).fetchall()
    for row in rows:
        if row["category"] in counts:
            counts[row["category"]] = row["count"]
    return total, counts

def chart_gradient(counts, total):
    colors = ["#d5a267", "#64aeb6", "#8495a9", "#e2bf62", "#8b82d4", "#d17e78"]
    if not total:
        return "#edf1ed 0deg 360deg"
    stops = []
    angle = 0.0
    for category, color in zip(CLASS_NAMES, colors):
        start = angle
        angle += counts[category] / total * 360
        if angle > start:
            stops.append(f"{color} {start:.2f}deg {angle:.2f}deg")
    return ", ".join(stops)


def is_allowed_image(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def preprocess_image(filepath):
    width, height = get_model_input_size(model)
    with Image.open(filepath) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_image(filepath):
    if model is None:
        raise RuntimeError('The trained classifier model is unavailable.')
    batch = preprocess_image(filepath)
    output = np.asarray(model.predict(batch, verbose=0))
    scores = output[0].reshape(-1)
    if scores.size != len(CLASS_NAMES):
        raise ValueError(
            f"The model returned {scores.size} scores; expected {len(CLASS_NAMES)} categories."
        )
    index = int(np.argmax(scores))
    return CLASS_NAMES[index], round(float(scores[index]) * 100, 2)


@app.errorhandler(413)
def too_large(_error):
    flash("The selected files are too large. Please keep the total upload under 64 MB.", "error")
    return redirect(url_for("home"))


@app.route("/", methods=["GET", "POST"])
def home():
    predictions = []
    if request.method == "POST":
        if model is None:
            flash(
                "The classifier could not start. Check that TensorFlow is installed and the existing model file is available.",
                "error",
            )
        else:
            files = [item for item in request.files.getlist("images") if item and item.filename]
            if not files:
                flash("Choose one or more image files to classify.", "error")
            elif len(files) > MAX_FILES_PER_BATCH:
                flash(f"Please upload no more than {MAX_FILES_PER_BATCH} images at a time.", "error")
            else:
                for uploaded in files:
                    safe_original = secure_filename(uploaded.filename or "")
                    if not safe_original or not is_allowed_image(safe_original):
                        flash(f"Skipped {uploaded.filename!r}: choose a supported image file.", "error")
                        continue

                    stored_name = f"{uuid.uuid4().hex}_{safe_original}"
                    filepath = UPLOAD_FOLDER / stored_name
                    try:
                        uploaded.save(filepath)
                        # Verify actual image content; extensions and browser MIME types are not trusted.
                        with Image.open(filepath) as image:
                            image.verify()
                        category, confidence = predict_image(filepath)
                        with connect_db() as db:
                            db.execute(
                                "INSERT INTO classifications (filename, original_name, category, confidence) VALUES (?, ?, ?, ?)",
                                (stored_name, safe_original, category, confidence),
                            )
                        predictions.append({
                            "filename": safe_original,
                            "prediction": category,
                            "display_prediction": DISPLAY_NAMES[category],
                            "confidence": confidence,
                            "image_path": stored_name,
                        })
                    except (UnidentifiedImageError, OSError, ValueError) as exc:
                        app.logger.info("Rejected invalid image %s: %s", safe_original, exc)
                        flash(f"Skipped {safe_original}: the file is not a readable image.", "error")
                        filepath.unlink(missing_ok=True)
                    except Exception:
                        app.logger.exception("Prediction failed for %s", safe_original)
                        flash(f"Could not classify {safe_original}. Check the model input configuration.", "error")
                        filepath.unlink(missing_ok=True)
                if predictions:
                    flash(f"Classified {len(predictions)} image(s) successfully.", "success")

    total_classified, category_counts = dashboard_stats()
    # All successfully stored, classified images are counted as uploads.
    return render_template(
        "index.html",
        predictions=predictions,
        category_counts=category_counts,
        chart_gradient=chart_gradient(category_counts, total_classified),
        total_uploaded=total_classified,
        total_classified=total_classified,
        display_names=DISPLAY_NAMES,
        model_ready=model is not None,
    )


if __name__ == "__main__":
    app.run(debug=False)


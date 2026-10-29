"""PhishGuard - Flask Backend Application.

Step 4: Backend development for AI/ML-Based Phishing Website Detection.

Endpoints:
- GET  /health   : Health-check verifying service status and model loading.
- POST /predict  : Accepts JSON with a URL, validates it, extracts available lexical
                   features, audits coverage against the 30 required model features,
                   and safely refuses prediction with a clear explanation if feature
                   extraction is incomplete. Can also accept full 30-feature payloads.

Educational / Prototype Boundary Notice:
This project is an academic prototype demonstrating machine learning pipeline principles.
It is NOT a production-ready security perimeter.
"""

from pathlib import Path
import sys
import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template

import url_feature_extractor as ufe

# Paths & Global State
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METADATA_FILE = MODELS_DIR / "model_metadata.joblib"
MODEL_FILE = MODELS_DIR / "phishing_rf_model.joblib"

app = Flask(__name__)

# Load model and metadata
model = None
metadata = None
expected_features = []
labels_map = {-1: "Phishing", 1: "Legitimate"}

try:
    if METADATA_FILE.exists():
        metadata = joblib.load(METADATA_FILE)
        model = metadata.get("model")
        expected_features = metadata.get("feature_names", [])
        labels_map = metadata.get("labels", labels_map)
        print(f"[INIT] Model & metadata loaded successfully. Features required: {len(expected_features)}")
    elif MODEL_FILE.exists():
        model = joblib.load(MODEL_FILE)
        print("[INIT] Model loaded directly from model file.")
    else:
        print("[WARNING] Model artifacts not found. Run train_model.py first.")
except Exception as exc:
    print(f"[FATAL ERROR] Failed to load model artifact: {exc}")

# Load the URL-only model
URL_MODEL_FILE = MODELS_DIR / "phishing_url_only_model.joblib"
URL_METADATA_FILE = MODELS_DIR / "url_only_model_metadata.joblib"

url_model = None
url_metadata = None

try:
    if URL_METADATA_FILE.exists():
        url_metadata = joblib.load(URL_METADATA_FILE)
        url_model = url_metadata.get("model")
    elif URL_MODEL_FILE.exists():
        url_model = joblib.load(URL_MODEL_FILE)
    print("[INIT] URL-only model loaded." if url_model is not None else "[WARNING] URL-only model not loaded.")
except Exception as exc:
    print(f"[WARNING] URL-only model loading failed: {exc}")


@app.route("/", methods=["GET"])
def index():
    """Render the PhishGuard frontend dashboard."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Service health and model status check."""
    is_ready = model is not None and len(expected_features) == 30
    response = {
        "status": "healthy" if is_ready else "degraded",
        "service": "PhishGuard AI Phishing Detection API",
        "version": "1.0.0-prototype",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features_required_count": len(expected_features),
        "disclaimer": "Academic prototype for research/education. Not a production security gateway."
    }
    status_code = 200 if is_ready else 503
    return jsonify(response), status_code


@app.route("/predict", methods=["POST"])
def predict():
    """Predict phishing vs legitimate website.

    Expects JSON body:
    1. Either `{"url": "https://example.com"}`:
       Extracts lexical features, compares against the 30 required model features,
       and safely refuses to predict if external/DOM features are missing (Requirement 6).
    2. Or `{"features": { ... 30 feature key-value pairs ... }}`:
       Executes model inference when all 30 features are explicitly supplied.
    """
    if model is None:
        return jsonify({
            "error": "Model not initialized",
            "message": "The Random Forest model is not currently loaded on the server."
        }), 503

    # Ensure JSON payload
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type",
            "message": "Request body must be valid JSON with Content-Type: application/json."
        }), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON format",
            "message": "Request payload must be a JSON object."
        }), 400

    # Path A: User submitted a raw URL
    if "url" in data:
        raw_url = data.get("url")
        is_valid, err_msg, parsed = ufe.validate_url(raw_url)
        if not is_valid:
            return jsonify({
                "error": "Invalid URL",
                "message": err_msg,
                "input_received": raw_url
            }), 400

        # Extract available lexical features (strictly from URL string)
        try:
            extracted_features = ufe.extract_lexical_features(raw_url)
        except Exception as exc:
            return jsonify({
                "error": "Feature extraction failure",
                "message": str(exc)
            }), 400

                # Predict using the URL-only model and its 8 trained features
        if url_model is None:
            return jsonify({
                "error": "URL model not initialized",
                "message": "The URL-only model could not be loaded."
            }), 503

        url_feature_names = url_metadata.get(
            "feature_names", list(extracted_features.keys())
        ) if url_metadata else list(extracted_features.keys())

        missing = [
            name for name in url_feature_names
            if name not in extracted_features
        ]

        if missing:
            return jsonify({
                "error": "Missing URL features",
                "missing_features": missing
            }), 422

        X_url = pd.DataFrame([{
            name: extracted_features[name]
            for name in url_feature_names
        }])

        pred = int(url_model.predict(X_url)[0])
        probabilities = url_model.predict_proba(X_url)[0]
        class_indices = {
            cls: idx for idx, cls in enumerate(url_model.classes_)
        }

        prob_phishing = float(probabilities[class_indices[-1]])
        prob_legitimate = float(probabilities[class_indices[1]])

        return jsonify({
            "status": "success",
            "prediction": pred,
            "label": "Phishing" if pred == -1 else "Legitimate",
            "confidence": round(
                max(prob_phishing, prob_legitimate) * 100, 2
            ),
            "probabilities": {
                "phishing": round(prob_phishing, 4),
                "legitimate": round(prob_legitimate, 4)
            },
            "features_used_count": len(url_feature_names),
            "notice": (
                "Educational URL-only estimate; not a definitive "
                "security verdict."
            )
        }), 200

    # Path B: User submitted a complete 30-feature vector
    elif "features" in data:
        features_dict = data.get("features")
        if not isinstance(features_dict, dict):
            return jsonify({
                "error": "Invalid features format",
                "message": "'features' must be an object with feature-name to value mappings."
            }), 400

        # Verify all 30 features are present
        missing = [f for f in expected_features if f not in features_dict]
        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_count": len(missing),
                "missing_features": missing
            }), 400

        # Construct single-row DataFrame in exact column order
        ordered_row = {f: [features_dict[f]] for f in expected_features}
        X_infer = pd.DataFrame(ordered_row)

        pred = int(model.predict(X_infer)[0])
        probabilities = model.predict_proba(X_infer)[0]
        # model.classes_ order
        class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}

        prob_phishing = float(probabilities[class_indices[-1]])
        prob_legitimate = float(probabilities[class_indices[1]])

        label_str = "Phishing" if pred == -1 else "Legitimate"

        return jsonify({
            "status": "success",
            "prediction": pred,
            "label": label_str,
            "confidence": round(max(prob_phishing, prob_legitimate) * 100, 2),
            "probabilities": {
                "phishing": round(prob_phishing, 4),
                "legitimate": round(prob_legitimate, 4)
            },
            "features_used_count": len(expected_features)
        }), 200

    else:
        return jsonify({
            "error": "Missing required field",
            "message": "Request JSON must contain either a 'url' string or a 30-item 'features' object."
        }), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": "The requested endpoint does not exist."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method Not Allowed", "message": "HTTP method not supported for this endpoint."}), 405


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal Server Error", "message": "An unexpected server error occurred."}), 500


if __name__ == "__main__":
    print("[SERVER] Starting PhishGuard backend service on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

"""
NLP Document Summarizer - Flask REST API
Provides endpoints for summarizing text and files.
"""

import os
import time
import tempfile
from flask import Flask, request, jsonify

from summarizer import DocumentSummarizer
from config import CONFIG


app = Flask(__name__)
summarizer = None


def get_summarizer():
    """Lazy-load the summarizer model."""
    global summarizer
    if summarizer is None:
        print("Initializing summarizer model...")
        summarizer = DocumentSummarizer()
    return summarizer


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": CONFIG["model_name"],
        "version": "1.0.0"
    })


@app.route("/summarize", methods=["POST"])
def summarize_text():
    """
    Summarize text provided in the request body.

    Request JSON:
        {
            "text": "Your document text...",
            "max_length": 150,     (optional)
            "min_length": 40,      (optional)
            "num_beams": 4         (optional)
        }

    Returns JSON:
        {
            "summary": "The generated summary...",
            "original_words": 2147,
            "summary_words": 89,
            "compression_ratio": 0.9585,
            "processing_time": 3.2,
            "chunks_used": 3
        }
    """
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"]
    if not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400

    # Optional parameters
    params = {}
    if "max_length" in data:
        params["max_length"] = int(data["max_length"])
    if "min_length" in data:
        params["min_length"] = int(data["min_length"])
    if "num_beams" in data:
        params["num_beams"] = int(data["num_beams"])
    if "length_penalty" in data:
        params["length_penalty"] = float(data["length_penalty"])

    try:
        model = get_summarizer()
        result = model.summarize(text, **params)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarize/file", methods=["POST"])
def summarize_file():
    """
    Summarize an uploaded text file.

    Form data:
        file: The text file to summarize
        max_length: (optional) Max summary tokens
        min_length: (optional) Min summary tokens
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Check file extension
    allowed_extensions = {".txt", ".md", ".html", ".text"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({
            "error": f"Unsupported file type: {ext}. "
                     f"Allowed: {', '.join(allowed_extensions)}"
        }), 400

    # Save to temp file and process
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as tmp:
            content = file.read().decode("utf-8")
            tmp.write(content)
            tmp_path = tmp.name

        # Get optional parameters from form data
        params = {}
        if "max_length" in request.form:
            params["max_length"] = int(request.form["max_length"])
        if "min_length" in request.form:
            params["min_length"] = int(request.form["min_length"])

        model = get_summarizer()
        result = model.summarize(content, **params)
        result["filename"] = file.filename

        return jsonify(result)

    except UnicodeDecodeError:
        return jsonify({"error": "Could not decode file. Please use UTF-8 encoding."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": {
            "POST /summarize": "Summarize text from JSON body",
            "POST /summarize/file": "Summarize an uploaded text file",
            "GET /health": "Health check"
        }
    }), 404


if __name__ == "__main__":
    # Pre-load the model on startup
    get_summarizer()

    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB limit
    app.run(
        host=CONFIG.get("api_host", "0.0.0.0"),
        port=CONFIG.get("api_port", 5000),
        debug=False
    )

"""
app.py
Flask web server for the Instagram Caption Generator.
Model and vocab are loaded ONCE at startup.
"""

from flask import Flask, render_template, request, jsonify
from model.infer import load_model, generate_caption

app = Flask(__name__)

# ── Load model at startup (avoids per-request lag) ────────────────────────────
load_model()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)

    segment     = data.get("segment", "").strip()
    description = data.get("description", "").strip()

    if not segment or not description:
        return jsonify({"error": "Both 'segment' and 'description' are required."}), 400

    caption = generate_caption(segment, description)
    return jsonify({"caption": caption})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
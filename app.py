"""Railway OMR Auto Evaluation System - web server.

  python app.py                      (development, http://localhost:8000)
  gunicorn -w 2 -t 180 app:app       (production)

Files are processed in memory only; nothing is written to disk or stored.
"""
import os
import traceback

from flask import Flask, jsonify, request, send_from_directory

from engine import keyparser, omr, scoring
from engine.fileio import FileProblem

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def _problem(code, message, status=200):
    resp = jsonify(ok=False, blocking=True, problems=[dict(level="error", code=code, message=message)])
    resp.status_code = status
    return resp


def _file(name):
    f = request.files.get(name)
    if f is None:
        return None
    data = f.read()
    return data or None


@app.errorhandler(413)
def too_big(_):
    return _problem("too_large", "फ़ाइल बहुत बड़ी है (अधिकतम 25 MB)।", 413)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/healthz")
def healthz():
    return jsonify(ok=True)


@app.post("/api/read-key")
def read_key():
    data = _file("file")
    if not data:
        return _problem("no_file", "उत्तर कुंजी की फ़ाइल नहीं मिली।")
    try:
        key = keyparser.read_answer_key(data)
    except FileProblem as e:
        return _problem(e.code, e.message)
    except Exception:
        traceback.print_exc()
        return _problem("key_failed", "उत्तर कुंजी पढ़ते समय त्रुटि हुई। कृपया साफ़ और पूरी फ़ाइल दें।", 500)
    key["ok"] = not key["blocking"]
    return jsonify(key)


@app.post("/api/read-omr")
def read_omr():
    data = _file("file")
    if not data:
        return _problem("no_file", "OMR शीट की फ़ाइल नहीं मिली।")
    force = request.form.get("force_layout") == "1"
    try:
        res = omr.read_omr(data, force_layout=force)
    except FileProblem as e:
        return _problem(e.code, e.message)
    except Exception:
        traceback.print_exc()
        return _problem("omr_failed", "OMR शीट पढ़ते समय त्रुटि हुई। कृपया साफ़ फ़ोटो/स्कैन दें।", 500)
    return jsonify(res)


@app.post("/api/score")
def score():
    body = request.get_json(silent=True) or {}
    res = scoring.calculate(body.get("key"), body.get("omr"), body.get("settings"))
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)

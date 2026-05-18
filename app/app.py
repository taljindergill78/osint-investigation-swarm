"""Flask demo app: run investigation from a web form and display results."""

from __future__ import annotations

import sys
import uuid as _uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")
import os as _os
print(f"[app] GROQ_API_KEY loaded: {bool(_os.environ.get('GROQ_API_KEY'))}", flush=True)

from flask import Flask, redirect, render_template, request, url_for

from app.pipeline import run_investigation, get_registered_entities, QUERY_TEMPLATES

app = Flask(__name__, template_folder=Path(__file__).resolve().parent / "templates")

_result_cache: dict = {}
_MAX_CACHED = 20


@app.route("/", methods=["GET"])
def index():
    entities = get_registered_entities()
    templates = QUERY_TEMPLATES
    return render_template("index.html", entities=entities, templates=templates)


@app.route("/healthz", methods=["GET"])
def healthz():
    return {"status": "ok"}, 200


@app.route("/", methods=["POST"])
def submit():
    entities = get_registered_entities()
    templates = QUERY_TEMPLATES
    query = (request.form.get("query") or "").strip()
    if not query:
        return render_template("index.html", error="Please enter an investigation query.", entities=entities, templates=templates)
    data_root = ROOT / "data"
    result = run_investigation(query, data_root=data_root)
    if result.get("report_html"):
        html = result["report_html"]
        if "<body>" in html and "</body>" in html:
            start = html.index("<body>") + len("<body>")
            end = html.index("</body>")
            result["report_body"] = html[start:end].strip()
        else:
            result["report_body"] = html
    else:
        result["report_body"] = ""
    result_id = str(_uuid.uuid4())
    _result_cache[result_id] = result
    if len(_result_cache) > _MAX_CACHED:
        oldest = next(iter(_result_cache))
        del _result_cache[oldest]
    return redirect(url_for("results_page", result_id=result_id))


@app.route("/results/<result_id>")
def results_page(result_id: str):
    result = _result_cache.get(result_id)
    if not result:
        return redirect(url_for("index"))
    return render_template("results.html", result=result)


def main():
    app.run(host="0.0.0.0", port=5001, debug=True)


if __name__ == "__main__":
    main()

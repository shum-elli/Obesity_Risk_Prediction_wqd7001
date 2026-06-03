import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "personal_obesity_bundle.joblib"
HOST = "127.0.0.1"
PORT = 8000

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Obesity Questionnaire Predictor</title>
  <style>
    body{margin:0;font-family:"Segoe UI",sans-serif;background:#f7f8fa;color:#18212a}
    .wrap{max-width:1180px;margin:0 auto;padding:24px}
    .hero,.main{display:grid;grid-template-columns:1.05fr .95fr;gap:20px}
    .card{background:#fff;border:1px solid #dde3ea;border-radius:8px;box-shadow:0 10px 26px rgba(24,33,42,.07)}
    .hero .card,.main .card{padding:24px}
    h1{font-size:44px;line-height:1;margin:12px 0}
    .lead,.muted{color:#5b6772;line-height:1.6}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#e7f0ff;color:#245aa8;font-size:12px;font-weight:700;text-transform:uppercase}
    .stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:18px}
    .results{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:18px}
    .results .box:first-child{grid-column:1/-1}
    .box{padding:14px;border-radius:8px;background:#f9fbfd;border:1px solid #dde3ea}
    .box strong{display:block;font-size:24px;line-height:1.15;overflow-wrap:anywhere;word-break:break-word}
    #topClass{font-size:clamp(18px,2.2vw,24px)}
    .heroResult{background:#17212b;color:#fff}
    .score{font-size:62px;font-weight:800;line-height:.95;margin:14px 0}
    .bar{height:12px;background:rgba(255,255,255,.15);border-radius:999px;overflow:hidden;margin-top:18px}
    .fill{height:100%;width:0;background:linear-gradient(90deg,#4caf70,#f2b544,#d84b3a)}
    form{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
    .field{display:flex;flex-direction:column;gap:6px}
    .full{grid-column:1/-1}
    label{font-weight:700;font-size:14px}
    input,select,button{padding:12px 13px;border-radius:8px;font-size:14px;font-family:inherit}
    input,select{border:1px solid #cfd8e3;background:#fff}
    button{border:none;background:#2563ad;color:#fff;font-weight:700;cursor:pointer}
    .hint,.status,.notes{font-size:13px;color:#5b6772;line-height:1.5}
    .status{min-height:20px;margin-top:12px;color:#9a4d31;font-weight:700}
    .probRow{display:grid;grid-template-columns:160px 1fr 60px;gap:10px;align-items:center;margin-top:10px;font-size:14px}
    .track{height:10px;background:#e4e9ef;border-radius:999px;overflow:hidden}
    .track > div{height:100%;background:#2563ad}
    .modelTable{width:100%;border-collapse:collapse;margin-top:14px;font-size:14px}
    .modelTable th,.modelTable td{padding:10px;border-bottom:1px solid #dde3ea;text-align:right}
    .modelTable th:first-child,.modelTable td:first-child{text-align:left}
    .modelTable th{color:#5b6772;font-weight:700;background:#f9fbfd}
    .bestTag{display:inline-block;margin-left:6px;padding:2px 6px;border-radius:999px;background:#e7f0ff;color:#245aa8;font-size:11px;font-weight:700}
    .disclaimer{margin-top:18px;padding:12px;border-left:4px solid #2563ad;background:#f3f7fc;color:#42505c;font-size:13px;line-height:1.5}
    @media (max-width: 900px){.hero,.main,form{grid-template-columns:1fr}.stats,.results{grid-template-columns:1fr}.full{grid-column:auto}}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="card">
        <span class="pill">GA2 Data Product</span>
        <h1>Questionnaire-based obesity risk prediction.</h1>
        <p class="lead">This web version mirrors the Group Assignment 2 notebook. It compares seven trained models, uses the best model by default, and predicts obesity-related risk from lifestyle questionnaire inputs.</p>
        <div style="margin-top: 16px;">
          <label for="modelSelect" style="display: block; font-weight: 700; font-size: 14px; margin-bottom: 6px;">Select Model:</label>
          <select id="modelSelect" style="padding: 12px 13px; border-radius: 12px; font-size: 14px; border: 1px solid #dfd3c8; background: #fff; width: 100%; cursor: pointer;"></select>
        </div>
        <div class="stats">
          <div class="box"><strong id="acc">--</strong><span class="muted">Accuracy</span></div>
          <div class="box"><strong id="f1">--</strong><span class="muted">Weighted F1</span></div>
          <div class="box"><strong id="auc">--</strong><span class="muted">Obesity AUC</span></div>
          <div class="box"><strong id="riskAuc">--</strong><span class="muted">At-risk AUC</span></div>
        </div>
        <div class="disclaimer">Inputs intentionally exclude height, weight, and BMI. The output is for coursework demonstration and risk awareness, not medical diagnosis.</div>
      </div>
      <div class="card heroResult">
        <div class="muted" style="color:rgba(255,255,255,.75)">Predicted obesity probability</div>
        <div class="score" id="risk">--%</div>
        <div id="riskText" style="color:rgba(255,255,255,.82)">Waiting for questionnaire input</div>
        <div class="bar"><div class="fill" id="riskFill"></div></div>
      </div>
    </section>

    <section class="main" style="margin-top:20px">
      <div class="card">
        <h2>Questionnaire</h2>
        <p class="muted">Numeric scores follow the original dataset scale.</p>
        <form id="form"></form>
        <div class="status" id="status"></div>
      </div>
      <div class="card">
        <h2>Prediction</h2>
        <div class="results">
          <div class="box"><strong id="topClass">--</strong><span class="muted">Most likely class</span></div>
          <div class="box"><strong id="topProb">--%</strong><span class="muted">Top class probability</span></div>
          <div class="box"><strong id="riskAt">--%</strong><span class="muted">Overweight/obesity risk</span></div>
        </div>
        <div id="probabilities" style="margin-top:18px"></div>
        <div class="notes" id="notes" style="margin-top:18px"></div>
      </div>
    </section>
    <section class="card" style="margin-top:20px;padding:24px">
      <h2>Model Comparison</h2>
      <p class="muted">Metrics are loaded from the latest trained model bundle generated by the notebook or training script.</p>
      <div id="modelTable"></div>
    </section>
  </div>
  <script>
    let meta = null;
    let selectedModel = null;
    const form = document.getElementById('form');
    const modelSelect = document.getElementById('modelSelect');

    function makeField(name, spec) {
      const wrap = document.createElement('div');
      wrap.className = 'field';
      const label = document.createElement('label');
      label.htmlFor = name;
      label.textContent = spec.label;
      wrap.appendChild(label);
      let el;
      if (spec.type === 'select') {
        el = document.createElement('select');
        spec.options.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v;
          opt.textContent = v;
          el.appendChild(opt);
        });
      } else {
        el = document.createElement('input');
        el.type = 'number';
        el.min = spec.min;
        el.max = spec.max;
        el.step = spec.step;
        el.value = spec.default;
      }
      el.id = name;
      wrap.appendChild(el);
      if (spec.hint) {
        const hint = document.createElement('div');
        hint.className = 'hint';
        hint.textContent = spec.hint;
        wrap.appendChild(hint);
      }
      form.appendChild(wrap);
    }

    function renderMetrics(m) {
      if (!selectedModel || !m.models[selectedModel]) return;
      const metrics = m.models[selectedModel].metrics;
      document.getElementById('acc').textContent = metrics.accuracy.toFixed(3);
      document.getElementById('f1').textContent = metrics.weighted_f1.toFixed(3);
      document.getElementById('auc').textContent = metrics.obesity_auc.toFixed(3);
      document.getElementById('riskAuc').textContent = metrics.overweight_or_obesity_auc.toFixed(3);
    }

    function renderModelTable(m) {
      const rows = Object.entries(m.models)
        .map(([name, info]) => ({ name, ...info.metrics }))
        .sort((a, b) => b.accuracy - a.accuracy);
      const table = document.createElement('table');
      table.className = 'modelTable';
      table.innerHTML = '<thead><tr><th>Model</th><th>Accuracy</th><th>Weighted F1</th><th>Obesity AUC</th><th>At-risk AUC</th></tr></thead>';
      const body = document.createElement('tbody');
      rows.forEach(row => {
        const tr = document.createElement('tr');
        const best = row.name === m.best_model ? '<span class="bestTag">Default</span>' : '';
        tr.innerHTML = `
          <td>${row.name}${best}</td>
          <td>${row.accuracy.toFixed(4)}</td>
          <td>${row.weighted_f1.toFixed(4)}</td>
          <td>${row.obesity_auc.toFixed(4)}</td>
          <td>${row.overweight_or_obesity_auc.toFixed(4)}</td>
        `;
        body.appendChild(tr);
      });
      table.appendChild(body);
      const node = document.getElementById('modelTable');
      node.innerHTML = '';
      node.appendChild(table);
    }

    function renderNotes(notes) {
      const node = document.getElementById('notes');
      node.innerHTML = '';
      notes.forEach(note => {
        const div = document.createElement('div');
        div.textContent = note;
        node.appendChild(div);
      });
    }

    function renderProbabilities(rows) {
      const node = document.getElementById('probabilities');
      node.innerHTML = '';
      rows.forEach(row => {
        const div = document.createElement('div');
        div.className = 'probRow';
        div.innerHTML = `<div>${row.label}</div><div class="track"><div style="width:${row.probability_percent.toFixed(2)}%"></div></div><div>${row.probability_percent.toFixed(1)}%</div>`;
        node.appendChild(div);
      });
    }

    async function init() {
      const res = await fetch('/api/meta');
      meta = await res.json();      
      // Initialize model selector with all available models
      Object.keys(meta.models).forEach(modelName => {
        const opt = document.createElement('option');
        opt.value = modelName;
        opt.textContent = modelName;
        if (modelName === meta.best_model) {
          opt.selected = true;
        }
        modelSelect.appendChild(opt);
      });
      
      selectedModel = modelSelect.value || meta.best_model;
      
      // Listen to model changes
      modelSelect.addEventListener('change', (e) => {
        selectedModel = e.target.value;
        renderMetrics(meta);
        document.getElementById('probabilities').innerHTML = '';
        document.getElementById('risk').textContent = '--%';
        document.getElementById('riskText').textContent = 'Waiting for questionnaire input';
        document.getElementById('riskFill').style.width = '0%';
        document.getElementById('topClass').textContent = '--';
        document.getElementById('topProb').textContent = '--%';
        document.getElementById('riskAt').textContent = '--%';
      });
      
      Object.entries(meta.feature_schema).forEach(([name, spec]) => makeField(name, spec));
      const submitWrap = document.createElement('div');
      submitWrap.className = 'field full';
      submitWrap.innerHTML = '<button type="submit">Predict From Questionnaire</button>';
      form.appendChild(submitWrap);
      renderMetrics(meta);
      renderModelTable(meta);
      renderNotes(meta.notes);
      form.addEventListener('submit', submitForm);
    }

    async function submitForm(event) {
      event.preventDefault();
      document.getElementById('status').textContent = 'Running prediction...';
      const payload = { model: selectedModel };
      Object.keys(meta.feature_schema).forEach(key => payload[key] = document.getElementById(key).value);
      const res = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      const data = await res.json();
      if (!res.ok) {
        document.getElementById('status').textContent = data.error || 'Prediction failed.';
        return;
      }
      document.getElementById('risk').textContent = `${data.obesity_probability_percent.toFixed(2)}%`;
      document.getElementById('riskText').textContent = `Most likely class: ${data.top_prediction.label}`;
      document.getElementById('riskFill').style.width = `${data.obesity_probability_percent.toFixed(2)}%`;
      document.getElementById('topClass').textContent = data.top_prediction.label;
      document.getElementById('topProb').textContent = `${data.top_prediction.probability_percent.toFixed(2)}%`;
      document.getElementById('riskAt').textContent = `${data.at_risk_probability_percent.toFixed(2)}%`;
      renderProbabilities(data.class_probabilities);
      renderNotes(data.notes);
      document.getElementById('status').textContent = 'Prediction updated.';
    }

    init().catch(err => document.getElementById('status').textContent = err.message);
  </script>
</body>
</html>
"""


def load_bundle() -> Dict[str, Any]:
    try:
        bundle = joblib.load(MODEL_PATH)
        # Support both old and new bundle formats for backward compatibility
        if "models" not in bundle:
            # Legacy format - wrap in new format
            classifier = bundle.get("classifier")
            bundle = {
                "models": {
                    "Default": {
                        "pipeline": classifier,
                        "metrics": bundle.get("metrics", {}).get("classifier", {}),
                    }
                },
                "best_model": "Default",
                "feature_columns": bundle.get("feature_columns", []),
                "classes": bundle.get("classes", []),
                "risk_classes": bundle.get("risk_classes", set()),
                "at_risk_classes": bundle.get("at_risk_classes", set()),
                "feature_schema": bundle.get("feature_schema", {}),
                "notes": bundle.get("notes", []),
            }
        return bundle
    except Exception as e:
        print(f"Error loading bundle: {e}")
        raise


def parse_payload(bundle: Dict[str, Any], payload: Dict[str, Any]) -> pd.DataFrame:
    row: Dict[str, Any] = {}
    for feature, spec in bundle["feature_schema"].items():
        value = payload.get(feature)
        if value is None or str(value).strip() == "":
            raise ValueError(f"Missing field: {feature}")
        row[feature] = float(value) if spec["type"] == "number" else str(value).strip()
    return pd.DataFrame([row], columns=bundle["feature_columns"])


def predict(bundle: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    model_name = payload.get("model", bundle.get("best_model"))
    if model_name not in bundle.get("models", {}):
        raise ValueError(f"Unknown model: {model_name}")
    
    sample = parse_payload(bundle, payload)
    pipeline = bundle["models"][model_name]["pipeline"]
    probabilities = pipeline.predict_proba(sample)[0]
    rows = []
    for label, prob in zip(bundle["classes"], probabilities):
        rows.append({"label": label, "probability": float(prob), "probability_percent": float(prob * 100.0)})
    rows.sort(key=lambda item: item["probability"], reverse=True)
    obesity_probability = sum(item["probability"] for item in rows if item["label"] in bundle["risk_classes"])
    at_risk_probability = sum(item["probability"] for item in rows if item["label"] in bundle["at_risk_classes"])
    return {
        "obesity_probability_percent": round(obesity_probability * 100.0, 4),
        "at_risk_probability_percent": round(at_risk_probability * 100.0, 4),
        "top_prediction": rows[0],
        "class_probabilities": rows,
        "notes": bundle["notes"],
    }


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/api/meta":
            bundle = load_bundle()
            meta_response = {
                "feature_schema": bundle["feature_schema"],
                "models": {
                    name: {
                        "metrics": model_data["metrics"]
                    }
                    for name, model_data in bundle.get("models", {}).items()
                },
                "best_model": bundle.get("best_model"),
                "notes": bundle["notes"],
            }
            self._send_json(meta_response)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            result = predict(load_bundle(), payload)
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("Model bundle not found. Run `python train_personal_obesity_models.py` first.")
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Web app running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

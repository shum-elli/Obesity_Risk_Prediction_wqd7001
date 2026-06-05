import json
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


WEB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEB_DIR.parent
if not (PROJECT_DIR / "personal_obesity_bundle.joblib").exists():
    PROJECT_DIR = WEB_DIR.parent / "project"
HOST = "127.0.0.1"
PORT = 8088

sys.path.insert(0, str(PROJECT_DIR))

from sklearn.impute import SimpleImputer  # noqa: E402
from web_app import load_bundle as project_load_bundle, predict  # noqa: E402


def load_bundle():
    bundle = project_load_bundle()
    patch_simple_imputers(bundle)
    return bundle


def patch_simple_imputers(bundle):
    visited = set()

    def walk(obj):
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if (
            isinstance(obj, SimpleImputer)
            and hasattr(obj, "statistics_")
            and not hasattr(obj, "_fill_dtype")
        ):
            obj._fill_dtype = obj.statistics_.dtype

        if isinstance(obj, dict):
            for value in obj.values():
                walk(value)
            return
        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                walk(value)
            return
        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                walk(value)

    walk(bundle)


class VisualizationHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/meta":
            bundle = load_bundle()
            report_path = PROJECT_DIR / "personal_model_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self._send_json(
                {
                    "rows": report.get("rows"),
                    "features_used": report.get("features_used", bundle["feature_columns"]),
                    "feature_schema": bundle["feature_schema"],
                    "models": {
                        name: {"metrics": model_data["metrics"]}
                        for name, model_data in bundle.get("models", {}).items()
                    },
                    "best_model": bundle.get("best_model"),
                    "notes": bundle["notes"],
                }
            )
            return
        if parsed.path == "/api/eda":
            eda_path = PROJECT_DIR / "eda_outputs" / "eda_summary.json"
            self._send_json(json.loads(eda_path.read_text(encoding="utf-8")))
            return
        if parsed.path == "/web":
            self.path = "/index.html"
        elif parsed.path.startswith("/web/"):
            self.path = parsed.path[4:]
        super().do_GET()

    def do_POST(self):
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

    def log_message(self, format, *args):
        return

    def _send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer((HOST, PORT), VisualizationHandler)
    print(f"Model-backed visualization running at http://{HOST}:{PORT}/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

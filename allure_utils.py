"""
allure_utils.py
Lets the Streamlit app (app.py) record Allure results directly, without
running pytest, and turn them into a viewable report:

  write_case_result()  -> drop one Allure result JSON (+ attachments) per
                           evaluated test case into allure-results/
  ensure_allure_cli()  -> find `allure` on PATH, or auto-download the
                           commandline tool into .tools/ on first use
  generate_report()    -> `allure generate` allure-results/ -> allure-report/
  serve_report()       -> serve allure-report/ over local HTTP (Allure's
                           report does an AJAX fetch for data/*.json, so it
                           must be viewed via http://, not file://)
  clear_results()      -> wipe allure-results/ + allure-report/ so the next
                           report only reflects newly-run evaluations
"""

import http.server
import json
import platform
import shutil
import socket
import subprocess
import threading
import time
import uuid
import zipfile

import requests
from pathlib import Path
from typing import Optional

ALLURE_VERSION = "2.29.0"
ALLURE_DOWNLOAD_URL = (
    f"https://github.com/allure-framework/allure2/releases/download/"
    f"{ALLURE_VERSION}/allure-{ALLURE_VERSION}.zip"
)
TOOLS_DIR = Path(__file__).parent / ".tools"

_servers = {}  # report_dir (str) -> (server, thread, port)


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "EXCELLENT"
    elif score >= 0.70:
        return "GOOD"
    elif score >= 0.50:
        return "MODERATE"
    else:
        return "POOR"


def ensure_allure_cli() -> str:
    """Return a path/command that runs the Allure CLI, downloading it if needed.

    Checks PATH first, then a previously-downloaded copy in .tools/, and
    only hits the network if neither is present.
    """
    on_path = shutil.which("allure")
    if on_path:
        return on_path

    bin_name = "allure.bat" if platform.system() == "Windows" else "allure"
    existing = list(TOOLS_DIR.glob(f"allure-*/bin/{bin_name}"))
    if existing:
        return str(existing[0])

    if not shutil.which("java"):
        raise RuntimeError(
            "Allure CLI needs a Java runtime (JRE 8+) to run, and none was found on PATH. "
            "Install a JRE (e.g. https://adoptium.net) and try again."
        )

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TOOLS_DIR / f"allure-{ALLURE_VERSION}.zip"
    with requests.get(ALLURE_DOWNLOAD_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(TOOLS_DIR)
    zip_path.unlink(missing_ok=True)

    cli_path = TOOLS_DIR / f"allure-{ALLURE_VERSION}" / "bin" / bin_name
    if platform.system() != "Windows":
        cli_path.chmod(0o755)
    if not cli_path.exists():
        raise RuntimeError(f"Allure CLI download succeeded but {cli_path} is missing.")
    return str(cli_path)


def write_case_result(
    results_dir,
    *,
    test_id: str,
    query: str,
    ground_truth: str,
    answer: str,
    contexts: list,
    metrics: dict,
    status: str = "passed",
    status_message: Optional[str] = None,
    story: str = "Streamlit Evaluation",
) -> None:
    """Write one Allure result (+ JSON attachments) for a single evaluated case."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    result_uuid = str(uuid.uuid4())
    now_ms = int(time.time() * 1000)

    attachments = []

    inputs_attachment_file = f"{uuid.uuid4()}-attachment.json"
    (results_dir / inputs_attachment_file).write_text(
        json.dumps(
            {
                "test_id": test_id,
                "query": query,
                "ground_truth": ground_truth,
                "generated_answer": answer,
                "contexts": contexts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    attachments.append({"name": "RAGAS Inputs", "source": inputs_attachment_file, "type": "application/json"})

    if status == "passed":
        scores_attachment_file = f"{uuid.uuid4()}-attachment.json"
        (results_dir / scores_attachment_file).write_text(
            json.dumps(
                {
                    "metrics": {k: round(v, 4) for k, v in metrics.items()},
                    "interpretation": {k: _score_label(v) for k, v in metrics.items()},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        attachments.append({"name": "RAGAS Scores", "source": scores_attachment_file, "type": "application/json"})

    parameters = [
        {"name": "Query", "value": query},
        {"name": "Ground Truth", "value": ground_truth},
        {"name": "Generated Answer", "value": answer},
        {"name": "Contexts", "value": " | ".join(contexts)},
    ]
    for name, value in metrics.items():
        parameters.append({"name": name, "value": f"{value:.4f} ({_score_label(value)})"})

    result = {
        "uuid": result_uuid,
        "historyId": test_id,
        "name": f"[{test_id}] {query[:80]}",
        "fullName": f"Streamlit Evaluation: {test_id}",
        "status": status,
        "stage": "finished",
        "start": now_ms,
        "stop": now_ms,
        "parameters": parameters,
        "attachments": attachments,
        "labels": [
            {"name": "epic", "value": "RAG Evaluation"},
            {"name": "feature", "value": "RAGAS Metrics"},
            {"name": "story", "value": story},
            {"name": "suite", "value": story},
        ],
    }
    if status_message:
        result["statusDetails"] = {"message": status_message}

    (results_dir / f"{result_uuid}-result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def clear_results(results_dir, report_dir=None) -> None:
    """Delete all accumulated Allure results (and the built report, if any) so the
    next run starts a fresh report instead of mixing in old test cases."""
    results_dir = Path(results_dir)
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if report_dir is not None:
        report_dir = Path(report_dir)
        if report_dir.exists():
            shutil.rmtree(report_dir)


def generate_report(results_dir, report_dir) -> None:
    """Build the static Allure HTML report from accumulated results."""
    allure_cli = ensure_allure_cli()
    subprocess.run(
        [allure_cli, "generate", str(results_dir), "--clean", "-o", str(report_dir)],
        check=True,
        capture_output=True,
        text=True,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_report(report_dir) -> str:
    """Serve an already-generated report dir over local HTTP, reusing a running
    server for the same directory across Streamlit reruns. Returns the URL."""
    key = str(Path(report_dir).resolve())

    existing = _servers.get(key)
    if existing is not None:
        server, _, port = existing
        return f"http://127.0.0.1:{port}/index.html"

    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
        *args, directory=key, **kwargs
    )
    port = _free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _servers[key] = (server, thread, port)
    return f"http://127.0.0.1:{port}/index.html"

"""Loopback-only browser UI. No external service, upload or remote access."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading
import webbrowser

from .core import DEFAULT_SOURCE, Limits, copy_approved, scan, write_reports


class Session:
    def __init__(self, resources, reports):
        self.resources, self.reports = Path(resources), Path(reports)
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.state = dict(
            status="idle",
            message="Choose the source and run stage 1. No dataset files are copied during scanning.",
            plan=None,
            reports="",
            copied=[],
        )

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def start(self, action):
        with self.lock:
            if self.state["status"] in ("scanning", "copying"):
                raise ValueError("A job is already running")
            self.cancel.clear()
            self.state.update(
                status="scanning" if action[0] == "scan" else "copying",
                message="Starting…",
            )
            if action[0] == "scan":
                self.state.update(plan=None, reports="", copied=[])
            thread = threading.Thread(target=self.run, args=(action,), daemon=True)
            thread.start()

    def run(self, action):
        try:

            def progress(message):
                self.update(message=message)

            if action[0] == "scan":
                _, source, settings = action
                plan = scan(
                    source,
                    self.resources / "reference_masses.csv",
                    Limits(**settings),
                    self.cancel,
                    progress,
                )
                self.reports.mkdir(parents=True, exist_ok=True)
                report = self.reports / (
                    datetime.now().strftime("%Y%m%d_%H%M%S_") + plan["scan_id"][:8]
                )
                write_reports(plan, report)
                self.update(
                    status="ready",
                    plan=plan,
                    reports=str(report),
                    message="Stage 1 complete. Review the results and selected size. Nothing has been copied.",
                )
            else:
                _, destination, ids, confirmed, scan_id = action
                plan = self.state["plan"]
                if not plan or plan["scan_id"] != scan_id:
                    raise ValueError("Scan plan changed; review the current plan first")
                completed = copy_approved(
                    plan,
                    destination,
                    ids,
                    confirm=confirmed,
                    cancel=self.cancel,
                    progress=progress,
                )
                self.update(
                    status="done",
                    copied=completed,
                    message=f"Stage 2 complete: {len(completed)} datasets copied and verified.",
                )
        except InterruptedError as exc:
            self.update(
                status="cancelled",
                message=str(exc)
                + ". Any .partial_ folders are incomplete; completed datasets and sources are preserved.",
            )
        except Exception as exc:
            self.update(
                status="error",
                message=f"{type(exc).__name__}: {exc}. Any .partial_ folders are incomplete and have been preserved.",
            )


def choose_folder():
    if os.name != "nt":
        raise ValueError(
            "The folder picker is available on Windows; type the path on other systems"
        )
    # No source/destination text is interpolated into executable code.
    script = "Add-Type -AssemblyName System.Windows.Forms; $picker = New-Object System.Windows.Forms.FolderBrowserDialog; $picker.Description = 'Choose folder for LASER dataset scanner'; if ($picker.ShowDialog() -eq 'OK') { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; [Console]::Write($picker.SelectedPath) }"
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
        capture_output=True,
        encoding="utf-8",
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return result.stdout.strip("\ufeff\r\n ")


def make_handler(session, token):
    prefix = "/" + token + "/"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def authorised(self):
            address = f"127.0.0.1:{self.server.server_port}"
            return (
                self.path.startswith(prefix)
                and self.headers.get("Host") == address
                and self.headers.get("Origin", "http://" + address)
                == "http://" + address
            )

        def do_GET(self):
            if not self.authorised():
                return self.respond(403, {"error": "Local scanner session only"})
            path = self.path[len(prefix) :]
            if path == "api/status":
                with session.lock:
                    payload = {
                        **session.state,
                        "default_source": DEFAULT_SOURCE,
                        "default_limits": asdict(Limits()),
                    }
                    return self.respond(200, payload)
            resources = {
                "": ("index.html", "text/html; charset=utf-8"),
                "app.js": ("app.js", "text/javascript; charset=utf-8"),
                "style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if path not in resources:
                return self.respond(404, {"error": "Not found"})
            name, mime = resources[path]
            return self.respond(
                200, (session.resources / "web" / name).read_bytes(), mime
            )

        def do_POST(self):
            if not self.authorised() or not secrets.compare_digest(
                self.headers.get("X-Scanner-Token", ""), token
            ):
                return self.respond(403, {"error": "Local scanner session only"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    not 0 < length <= 32768
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError("Invalid request")
                body = json.loads(self.rfile.read(length))
                path = self.path[len(prefix) :]
                if path == "api/pick-folder":
                    return self.respond(200, {"path": choose_folder()})
                if path == "api/scan":
                    session.start(("scan", str(body["source"]), body.get("limits", {})))
                elif path == "api/copy":
                    session.start(
                        (
                            "copy",
                            str(body["destination"]),
                            body["dataset_ids"],
                            body.get("confirm") is True,
                            body["scan_id"],
                        )
                    )
                elif path == "api/cancel":
                    session.cancel.set()
                elif path == "api/stop":
                    session.cancel.set()
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    return self.respond(404, {"error": "Not found"})
                return self.respond(200, {"ok": True})
            except (
                ValueError,
                KeyError,
                TypeError,
                OSError,
                subprocess.SubprocessError,
            ) as exc:
                return self.respond(400, {"error": str(exc)})

    return Handler


def serve(resources, reports, *, open_browser=True):
    session = Session(resources, reports)
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session, token))
    url = f"http://127.0.0.1:{server.server_port}/{token}/"
    print("LASER dataset scanner. Keep this window open while scanning/copying.")
    print("Open:", url, flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        session.cancel.set()
        server.server_close()

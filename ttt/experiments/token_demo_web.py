"""Local browser UI for the token demo. The gallery needs no GPU; pasted text uses the deployed app.

    python -m ttt.experiments.token_demo_web [--port 8766]
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

from ttt.experiments.token_demo_gallery_v1 import OUT as GALLERY
from ttt.experiments.token_demo_v1 import CHECKPOINT, FLAGS

PAGE = Path(__file__).with_name("token_demo.html")


def page() -> bytes:
    """The gallery is inlined, so the page also works saved as a single file."""
    return PAGE.read_text().replace("/*GALLERY*/null", GALLERY.read_text()).encode()


def handler_for(engine_factory):
    lock = threading.Lock()
    engine = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                return self._raw(200, "text/html; charset=utf-8", page())
            if self.path == "/gallery.json":
                return self._raw(200, "application/json", GALLERY.read_bytes())
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/score":
                return self._send(404, {"error": "not found"})
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            try:
                with lock:
                    if "app" not in engine:
                        engine["app"] = engine_factory()
                    result = engine["app"].score.remote(prime=payload["prime"], target=payload["target"])
            except Exception as error:
                return self._send(500, {"error": f"{type(error).__name__}: {error}"})
            self._send(200, result)

        def _send(self, status: int, body: dict):
            self._raw(status, "application/json", json.dumps(body).encode())

        def _raw(self, status: int, kind: str, data: bytes):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    return Handler


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="token_demo_web")
    parser.add_argument("--resume-from", default=CHECKPOINT)
    parser.add_argument("--flags", default=FLAGS)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    def engine():
        import modal

        return modal.Cls.from_name("ttt-token-demo-v1", "TokenDemo")(resume_from=args.resume_from, flags=args.flags)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(engine))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"token demo on {url}  (pasted text waits for a GPU container the first time, about a minute)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

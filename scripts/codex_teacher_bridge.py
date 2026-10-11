"""Bounded localhost transport between a GPU collector and the user's Codex CLI.

For a remote simulator, forward this loopback port through SSH. Only the temporary
bridge token travels to that machine; Codex login stays on the local computer.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
from pathlib import Path
import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from codex_collection_teacher import CodexTeacher, request_identity, validate_decision

MAX_BODY = 8 * 1024 * 1024


def read_token(path):
    token = Path(path).read_text().strip()
    if len(token) < 32 or any(c.isspace() for c in token):
        raise ValueError("Bridge token must contain at least 32 non-whitespace characters")
    return token


class RemoteTeacher:
    def __init__(self, url, token_file, *, timeout_seconds=150):
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Use a loopback HTTP URL, with an SSH tunnel for remote hosts")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Bridge URL must not contain credentials, query, or fragment")
        if timeout_seconds <= 0:
            raise ValueError("Teacher timeout must be positive")
        self.url = url.rstrip("/") + "/choose"
        self.token = read_token(token_file)
        self.timeout_seconds = timeout_seconds

    def choose(self, request, image_paths, output_dir):
        identity = request_identity(request, image_paths)
        body = json.dumps({"request": request, "images": [
            base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in image_paths
        ]}, allow_nan=False).encode()
        if len(body) > MAX_BODY:
            raise ValueError("Teacher snapshot is too large")
        req = Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.token,
        })
        # No retries: a timed-out request may already have consumed a decision.
        with build_opener(ProxyHandler({})).open(req, timeout=self.timeout_seconds) as response:
            answer = json.loads(response.read(64 * 1024 + 1))
        answer = validate_decision(answer, identity, request.get("collection_mode", "teacher_conditioned"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "teacher_response.json").open("x") as stream:
            json.dump(answer, stream, ensure_ascii=False, indent=2)
        return answer


def make_server(port, token, teacher, evidence_root, max_decisions, *, max_seconds=None):
    """Single-threaded to prevent overlapping Codex requests on this bridge."""
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("A positive decision budget is required")
    if max_seconds is not None and (type(max_seconds) is not int or max_seconds < 1):
        raise ValueError("A positive total time budget is required")
    evidence_root = Path(evidence_root)
    evidence_root.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log auth headers, observations, or server-side exceptions.

        def setup(self):
            super().setup()
            remaining = self.server.deadline - time.monotonic()
            self.connection.settimeout(max(0.001, min(15, remaining)))

        def send_json(self, status, value):
            encoded = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                return self.send_json(401, {"error": "unauthorized"})
            if self.path != "/choose":
                return self.send_json(404, {"error": "unknown endpoint"})
            if self.server.decisions >= max_decisions:
                return self.send_json(429, {"error": "decision budget exhausted"})
            if time.monotonic() >= self.server.deadline:
                return self.send_json(429, {"error": "time budget exhausted"})
            request_dir = None
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("invalid body size")
                payload = json.loads(self.rfile.read(size))
                if set(payload) != {"request", "images"} or not 3 <= len(payload["images"]) <= 5:
                    raise ValueError("three current and at most two context images are required")
                # Unique transport evidence; teacher's hash also rejects repeated requests.
                request_dir = evidence_root / ("incoming-" + secrets.token_hex(12))
                request_dir.mkdir()
                images = []
                for index, encoded in enumerate(payload["images"]):
                    image = request_dir / f"camera-{index}.png"
                    image.write_bytes(base64.b64decode(encoded, validate=True))
                    images.append(image)
                request_identity(payload["request"], images)
            except (ValueError, TypeError, KeyError, TimeoutError):
                return self.send_json(400, {"error": "invalid collection snapshot"})
            remaining = self.server.deadline - time.monotonic()
            if remaining <= 0:
                return self.send_json(429, {"error": "time budget exhausted"})
            self.server.decisions += 1
            previous_timeout = teacher.timeout_seconds
            try:
                teacher.timeout_seconds = min(previous_timeout, remaining)
                answer = teacher.choose(payload["request"], images, evidence_root / "decisions")
                self.send_json(200, answer)
            except Exception as error:
                (request_dir / "bridge_failure.json").write_text(json.dumps({
                    "error_type": type(error).__name__, "decision_charged_to_budget": True,
                }) + "\n")
                self.send_json(502, {"error": "teacher failed; inspect local evidence; no retry performed"})
            finally:
                teacher.timeout_seconds = previous_timeout

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.decisions = 0
    server.deadline = time.monotonic() + max_seconds if max_seconds is not None else float("inf")
    server.timeout = 1
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--decision-timeout", type=int, default=120)
    args = parser.parse_args(argv)
    if args.max_seconds < 1 or args.decision_timeout < 1:
        parser.error("Time budgets must be positive")
    if not args.token_file.exists():
        args.token_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    token = read_token(args.token_file)
    teacher = CodexTeacher(timeout_seconds=args.decision_timeout)
    server = make_server(args.port, token, teacher, args.output, args.max_decisions, max_seconds=args.max_seconds)
    print(json.dumps({"listening": f"http://127.0.0.1:{server.server_port}",
                      "max_decisions": args.max_decisions, "max_seconds": args.max_seconds}), flush=True)
    try:
        while time.monotonic() < server.deadline and server.decisions < args.max_decisions:
            server.handle_request()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

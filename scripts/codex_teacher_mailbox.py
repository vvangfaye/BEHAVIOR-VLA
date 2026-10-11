"""Finite private-AOSS mailbox between a GPU collector and the local Codex login.

The worker receives only two object-specific, expiring URLs. The local relay is
serial, preserves request evidence, and reserves identities before inference so
an interrupted or failed decision is never automatically sent to Codex again.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from codex_collection_teacher import (
    CodexTeacher, _inputs, _reject_duplicate_keys, request_identity,
    validate_decision as _teacher_validate_decision,
)

MAX_BODY = 8 * 1024 * 1024
MAX_RESPONSE = 64 * 1024
BUCKET = "260010165"
PREFIX = "behavior1k/codex-pilot-20261011/"
ID_RE = re.compile(r"^[0-9a-f]{64}$")


class MailboxError(RuntimeError):
    """Sanitized transport error; signed URLs must never reach logs."""


def _json(data):
    return json.loads(data, object_pairs_hook=_reject_duplicate_keys,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()


def _save(path, value):
    path = Path(path)
    path.write_bytes(_encode(value) + b"\n")


def _positive(value, limit, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= limit:
        raise ValueError(f"{name} must be positive and at most {limit}")
    return float(value)


def _validate_url(value):
    if not isinstance(value, str):
        raise ValueError("Mailbox URLs must be HTTPS presigned AOSS object URLs")
    parsed = urlparse(value)
    host = parsed.hostname or ""
    query = parse_qs(parsed.query)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment
            or parsed.port not in (None, 443)
            or not (host == "aoss.cn-sh-01b.sensecoreapi-oss.cn" or host == f"{BUCKET}.aoss.cn-sh-01b.sensecoreapi-oss.cn")
            or not (query.get("X-Amz-Signature") or query.get("Signature"))):
        raise ValueError("Mailbox URLs must be HTTPS presigned AOSS object URLs")
    _validate_key(_url_key(value))
    return value


def _url_key(value):
    parsed = urlparse(value)
    path = unquote(parsed.path)
    prefix = f"/{BUCKET}/" if parsed.hostname == "aoss.cn-sh-01b.sensecoreapi-oss.cn" else "/"
    if not path.startswith(prefix):
        raise ValueError("Mailbox URL must address the authorized bucket")
    return path[len(prefix):]


def _validate_key(key):
    if (not isinstance(key, str) or not key.startswith(PREFIX) or key == PREFIX
            or any(segment in ("", ".", "..") for segment in key.split("/"))
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", key)):
        raise ValueError("Mailbox object keys must remain inside the authorized pilot prefix")
    return key


def validate_decision(value, identity, collection_mode="teacher_conditioned"):
    """Reuse teacher validation, including its evolving correction schema."""
    if not isinstance(value, dict) or value.get("request_id") != identity or "response_id" in value:
        raise MailboxError("Teacher response identity or envelope is invalid")
    try:
        return _teacher_validate_decision(value, identity, collection_mode)
    except (ValueError, TypeError, KeyError):
        raise MailboxError("Teacher response does not match the decision contract") from None


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise MailboxError("Mailbox redirects are forbidden")


class WorkerAossTeacher:
    def __init__(self, config, *, opener=None, clock=time.monotonic, sleeper=time.sleep):
        try:
            config = _json(Path(config).read_bytes()) if isinstance(config, (str, Path)) else dict(config)
            if set(config) - {"request_put", "response_get", "timeout_seconds", "poll_seconds"}:
                raise ValueError("Unexpected mailbox configuration fields")
            self._request_put = _validate_url(config["request_put"])
            self._response_get = _validate_url(config["response_get"])
            if _url_key(self._request_put) == _url_key(self._response_get):
                raise ValueError("Request and response must use separate object URLs")
            self.timeout_seconds = _positive(config.get("timeout_seconds", 150), 180, "timeout_seconds")
            self.poll_seconds = _positive(config.get("poll_seconds", 1), 5, "poll_seconds")
        except (ValueError, TypeError, KeyError, OSError):
            raise ValueError("Invalid private mailbox configuration") from None
        self.opener = opener or build_opener(ProxyHandler({}), _NoRedirects())
        self.clock, self.sleeper = clock, sleeper

    def choose(self, request, image_paths, output_dir):
        # The collector is a POSIX main-thread process. Bound a slow HTTP body
        # as well as connect/read socket timeouts; preserve its outer deadline.
        from collect_b1k_codex import time_limit
        with time_limit(self.timeout_seconds):
            return self._choose(request, image_paths, output_dir)

    def _choose(self, request, image_paths, output_dir):
        frozen, images, identity = _inputs(request, image_paths)
        body = _encode({"request": frozen, "request_id": identity,
                        "images": [base64.b64encode(data).decode("ascii") for data in images]})
        if len(body) > MAX_BODY:
            raise ValueError("Teacher snapshot exceeds the 8 MiB mailbox limit")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Reserve before PUT: a timed-out upload may have arrived, so never reissue it.
        with (output_dir / "mailbox_request.json").open("x") as stream:
            json.dump({"request_id": identity, "status": "reserved", "transport": "private_aoss"}, stream)
        deadline = self.clock() + self.timeout_seconds
        try:
            req = Request(self._request_put, data=body, method="PUT", headers={"Content-Type": "application/json"})
            with self.opener.open(req, timeout=min(30, self.timeout_seconds)) as response:
                if response.status not in (200, 201, 204):
                    raise MailboxError("Mailbox upload failed; no retry performed")
            while self.clock() < deadline:
                remaining = deadline-self.clock()
                try:
                    req = Request(self._response_get, headers={"Cache-Control": "no-cache"})
                    with self.opener.open(req, timeout=min(15, remaining)) as response:
                        data = response.read(MAX_RESPONSE+1)
                    if len(data) > MAX_RESPONSE:
                        raise MailboxError("Mailbox response exceeds its size limit")
                    answer = _json(data)
                except HTTPError as error:
                    if error.code != 404:
                        raise MailboxError("Mailbox response download failed") from None
                    answer = None
                if isinstance(answer, dict) and answer.get("request_id") == identity:
                    if set(answer) == {"request_id", "error"}:
                        raise MailboxError("Local teacher failed; inspect local relay evidence; no retry performed")
                    decision = validate_decision(answer, identity, request.get("collection_mode", "teacher_conditioned"))
                    _save(output_dir / "teacher_response.json", decision)
                    return decision
                if answer is not None and (not isinstance(answer, dict) or not ID_RE.fullmatch(str(answer.get("request_id", "")))):
                    raise MailboxError("Malformed mailbox response")
                remaining = deadline-self.clock()
                if remaining > 0:
                    self.sleeper(min(self.poll_seconds, remaining))
            raise MailboxError("Mailbox teacher decision timed out; no retry performed")
        except Exception as error:
            # Do not interpolate exception text: urllib embeds signed URLs in it.
            _save(output_dir / "mailbox_failure.json", {"request_id": identity,
                                                       "error_type": type(error).__name__, "no_retry": True})
            if isinstance(error, MailboxError):
                raise
            raise MailboxError("Mailbox transport failed; no retry performed") from None


def _missing_object(error):
    response = getattr(error, "response", {})
    return response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound")


class AossMailboxRelay:
    def __init__(self, client, teacher, *, request_key, response_key, evidence_root,
                 max_decisions, max_seconds, health_key=None, clock=time.monotonic, sleeper=time.sleep):
        self.request_key, self.response_key = _validate_key(request_key), _validate_key(response_key)
        self.health_key = _validate_key(health_key or response_key + ".health.json")
        if len({self.request_key, self.response_key, self.health_key}) != 3:
            raise ValueError("Request, response and health require distinct objects")
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError("A finite positive integer decision budget is required")
        self.max_seconds = _positive(max_seconds, 86400, "max_seconds")
        self.client, self.teacher, self.max_decisions = client, teacher, max_decisions
        self.output = Path(evidence_root)
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.output / "incoming").mkdir(exist_ok=True, mode=0o700)
        (self.output / "reservations").mkdir(exist_ok=True, mode=0o700)
        self.clock, self.sleeper = clock, sleeper
        self.decisions = sum(path.is_dir() and bool(ID_RE.fullmatch(path.name))
                             for path in (self.output / "reservations").iterdir())
        self.deadline = self.clock() + self.max_seconds
        self._last_body_hash = None

    def _put(self, key, value):
        body = _encode(value)
        if len(body) > MAX_RESPONSE:
            raise MailboxError("Outgoing relay response is too large")
        try:
            self.client.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/json")
        except Exception:
            raise MailboxError("Cannot write private mailbox response") from None

    def health(self, status):
        value = {"contract": "codex_private_aoss_mailbox_v1", "status": status,
                 "ready": status in ("ready", "running"), "decisions": self.decisions,
                 "max_decisions": self.max_decisions, "seconds_remaining": max(0, self.deadline-self.clock())}
        _save(self.output / "health.json", value)
        self._put(self.health_key, value)
        return value

    def process_once(self):
        if self.decisions >= self.max_decisions or self.clock() >= self.deadline:
            return False
        try:
            response = self.client.get_object(Bucket=BUCKET, Key=self.request_key)
        except Exception as error:
            if _missing_object(error):
                return False
            raise MailboxError("Cannot read private mailbox request") from None
        stream = response["Body"]
        try:
            raw = stream.read(MAX_BODY+1)
        except Exception:
            raise MailboxError("Cannot read private mailbox request body") from None
        finally:
            stream.close()
        body_hash = hashlib.sha256(raw).hexdigest()
        if body_hash == self._last_body_hash:
            return False
        self._last_body_hash = body_hash
        incoming = self.output / "incoming" / body_hash
        incoming.mkdir(exist_ok=True, mode=0o700)
        (incoming / "payload.json").write_bytes(raw)
        identity = None
        try:
            if len(raw) > MAX_BODY:
                raise ValueError("Snapshot exceeds mailbox limit")
            payload = _json(raw)
            if not isinstance(payload, dict) or set(payload) != {"request", "images", "request_id"}:
                raise ValueError("Malformed mailbox request")
            claimed = payload["request_id"]
            if isinstance(claimed, str) and ID_RE.fullmatch(claimed):
                identity = claimed
            if (identity is None or not isinstance(payload["images"], list)
                    or not 3 <= len(payload["images"]) <= 5):
                raise ValueError("Expected an identity and 3–5 PNG snapshots")
            paths = []
            for index, value in enumerate(payload["images"]):
                path = incoming / f"image-{index}.png"
                path.write_bytes(base64.b64decode(value, validate=True))
                paths.append(path)
            if request_identity(payload["request"], paths) != identity:
                raise ValueError("Request identity mismatch")
        except (ValueError, TypeError, KeyError):
            _save(incoming / "failure.json", {"error": "invalid_snapshot", "no_retry": True})
            if identity:
                self._put(self.response_key, {"request_id": identity, "error": "invalid_snapshot"})
            return False
        reservation = self.output / "reservations" / identity
        try:
            reservation.mkdir(mode=0o700)
        except FileExistsError:
            # A completed exact request is replayed, never recomputed. In-flight
            # and failed reservations abort, including after a process restart.
            saved = reservation / "response.json"
            answer = _json(saved.read_bytes()) if saved.is_file() else {"request_id": identity, "error": "reserved_request_not_retried"}
            self._put(self.response_key, answer)
            return False
        _save(reservation / "request.json", {"request_id": identity, "incoming_sha256": body_hash,
                                              "status": "reserved", "no_retry": True})
        self.decisions += 1
        self.health("running")
        previous_timeout = self.teacher.timeout_seconds
        try:
            remaining = self.deadline-self.clock()
            if remaining <= 0:
                raise TimeoutError("Relay decision deadline elapsed")
            self.teacher.timeout_seconds = min(previous_timeout, remaining, 180)
            answer = self.teacher.choose(payload["request"], paths, self.output / "decisions")
            answer = validate_decision(answer, identity, payload["request"].get("collection_mode", "teacher_conditioned"))
        except Exception as error:
            _save(reservation / "failure.json", {"error_type": type(error).__name__, "no_retry": True})
            answer = {"request_id": identity, "error": "teacher_failed"}
        finally:
            self.teacher.timeout_seconds = previous_timeout
        _save(reservation / "response.json", answer)
        self._put(self.response_key, answer)
        return True

    def run(self):
        print(json.dumps(self.health("ready")), flush=True)
        try:
            while self.decisions < self.max_decisions and self.clock() < self.deadline:
                self.process_once()
                remaining = self.deadline-self.clock()
                if self.decisions < self.max_decisions and remaining > 0:
                    self.sleeper(min(1, remaining))
            return self.health("completed")
        except Exception:
            # No exception detail is published because SDK errors may include URLs.
            _save(self.output / "health.json", {"status": "failed", "ready": False, "decisions": self.decisions})
            raise MailboxError("Private mailbox relay failed; inspect local evidence") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-key", required=True)
    parser.add_argument("--response-key", required=True)
    parser.add_argument("--health-key")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--decision-timeout", type=int, default=120)
    args = parser.parse_args(argv)
    _positive(args.decision_timeout, 180, "decision_timeout")
    _validate_key(args.request_key)
    _validate_key(args.response_key)
    if args.health_key:
        _validate_key(args.health_key)
    from behavior_aoss_client import client
    relay = AossMailboxRelay(client(), CodexTeacher(timeout_seconds=args.decision_timeout),
        request_key=args.request_key, response_key=args.response_key, health_key=args.health_key,
        evidence_root=args.output, max_decisions=args.max_decisions, max_seconds=args.max_seconds)
    from collect_b1k_codex import time_limit
    with time_limit(args.max_seconds):
        print(json.dumps(relay.run()), flush=True)


if __name__ == "__main__":
    main()

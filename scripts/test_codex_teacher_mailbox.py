"""Private mailbox tests with in-memory fake S3/HTTP; no cloud or model calls."""
import base64
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from codex_collection_teacher import PNG_SIGNATURE, request_identity
from codex_teacher_mailbox import (
    AossMailboxRelay, BUCKET, MAX_BODY, MailboxError, PREFIX, WorkerAossTeacher,
)


class Clock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def fixture(tmp_path, count=3):
    request = {"task": "turning_on_radio", "instance_id": 311, "attempt_id": "fixture-only",
               "step": 0, "previous_subtask": None}
    paths = []
    for index in range(count):
        path = tmp_path / f"fixture-{index}.png"
        path.write_bytes(PNG_SIGNATURE + b"fixture-only" + bytes([index]))
        paths.append(path)
    if count > 3:
        request["context_images"] = ["demonstration" for _ in range(count-3)]
    return request, paths


def payload(request, paths):
    return {"request": request, "request_id": request_identity(request, paths),
            "images": [base64.b64encode(path.read_bytes()).decode() for path in paths]}


def decision(identity, **kwargs):
    return {"request_id": identity, "action": "continue", "subtask": "turn on the radio",
            "reason": "fixture-only", "correction": None, **kwargs}


def config(**kwargs):
    base = "https://aoss.cn-sh-01b.sensecoreapi-oss.cn/260010165/" + PREFIX
    return {"request_put": base + "fixture/request.json?X-Amz-Signature=SECRET-PUT",
            "response_get": base + "fixture/response.json?X-Amz-Signature=SECRET-GET",
            "timeout_seconds": 5, "poll_seconds": 1, **kwargs}


class HTTPResponse(io.BytesIO):
    status = 200


class Opener:
    def __init__(self, answers, *, fail_put=False):
        self.answers, self.calls, self.fail_put = list(answers), [], fail_put
    def open(self, request, timeout):
        self.calls.append((request.get_method(), request.data, timeout))
        if request.get_method() == "PUT":
            if self.fail_put:
                raise OSError("SECRET-PUT signed URL transport failed")
            return HTTPResponse(b"")
        value = self.answers.pop(0) if self.answers else None
        if value is None:
            raise HTTPError(request.full_url, 404, "not ready", {}, None)
        if isinstance(value, bytes):
            return HTTPResponse(value)
        return HTTPResponse(json.dumps(value).encode())


class NoSuchKey(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class S3:
    def __init__(self):
        self.objects, self.puts = {}, []
    def put_object(self, *, Bucket, Key, Body, ContentType):
        assert Bucket == BUCKET and ContentType == "application/json"
        self.objects[Key] = Body
        self.puts.append(Key)
    def get_object(self, *, Bucket, Key):
        assert Bucket == BUCKET
        if Key not in self.objects:
            raise NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}


class Teacher:
    timeout_seconds = 120
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail
    def choose(self, request, paths, output):
        self.calls.append((request, len(paths), self.timeout_seconds))
        if self.fail:
            raise RuntimeError("private underlying exception SECRET")
        return decision(request_identity(request, paths))


def relay(tmp_path, s3=None, teacher=None, clock=None, **kwargs):
    clock = clock or Clock()
    return AossMailboxRelay(s3 or S3(), teacher or Teacher(), request_key=PREFIX+"fixture/request.json",
        response_key=PREFIX+"fixture/response.json", evidence_root=tmp_path / "relay",
        max_decisions=kwargs.pop("max_decisions", 3), max_seconds=kwargs.pop("max_seconds", 10),
        clock=clock, sleeper=clock.sleep, **kwargs)


def test_worker_sends_once_ignores_stale_and_validates_matching_response(tmp_path):
    request, paths = fixture(tmp_path)
    identity = request_identity(request, paths)
    opener = Opener([None, decision("a"*64), decision(identity)])
    clock = Clock()
    worker = WorkerAossTeacher(config(), opener=opener, clock=clock, sleeper=clock.sleep)
    out = tmp_path / "worker"
    result = worker.choose(request, paths, out)
    assert result == decision(identity)
    assert [call[0] for call in opener.calls] == ["PUT", "GET", "GET", "GET"]
    uploaded = json.loads(opener.calls[0][1])
    assert uploaded == payload(request, paths)
    assert clock.now == 2
    assert json.loads((out / "teacher_response.json").read_text()) == result
    assert all("SECRET" not in file.read_text() for file in out.iterdir())
    with pytest.raises(FileExistsError):
        worker.choose(request, paths, out)
    assert len(opener.calls) == 4


@pytest.mark.parametrize("native", [True, False])
def test_mailbox_empty_continue_uses_the_request_mode(tmp_path, native):
    request, paths = fixture(tmp_path)
    if native:
        request.update(collection_mode="native_with_corrections", step=320,
                       current_native_subtask="reach the shoes", current_native_subtask_source="cached")
    identity = request_identity(request, paths)
    opener, clock = Opener([decision(identity, subtask="")]), Clock()
    worker = WorkerAossTeacher(config(), opener=opener, clock=clock, sleeper=clock.sleep)
    if native:
        assert worker.choose(request, paths, tmp_path / "worker")["subtask"] == ""
    else:
        with pytest.raises(MailboxError): worker.choose(request, paths, tmp_path / "worker")


def test_relay_validates_empty_continue_against_native_request(tmp_path):
    request, paths = fixture(tmp_path)
    request.update(collection_mode="native_with_corrections", step=256,
                   current_native_subtask="reach the radio", current_native_subtask_source="generated")
    s3, teacher = S3(), Teacher()
    teacher.choose = lambda req, imgs, output: decision(request_identity(req, imgs), subtask="")
    s3.objects[PREFIX + "fixture/request.json"] = json.dumps(payload(request, paths)).encode()
    actual = relay(tmp_path, s3=s3, teacher=teacher)
    assert actual.process_once()
    response = json.loads(s3.objects[PREFIX + "fixture/response.json"])
    assert response["action"] == "continue" and response["subtask"] == ""


def test_worker_timeout_does_not_reput_or_expose_signed_url(tmp_path):
    request, paths = fixture(tmp_path)
    opener, clock = Opener([]), Clock()
    worker = WorkerAossTeacher(config(), opener=opener, clock=clock, sleeper=clock.sleep)
    with pytest.raises(MailboxError, match="timed out") as caught:
        worker.choose(request, paths, tmp_path / "worker")
    assert clock.now == 5
    assert sum(call[0] == "PUT" for call in opener.calls) == 1
    assert "SECRET" not in str(caught.value)


@pytest.mark.parametrize("answer", [
    {"action": "success"}, {"subtask": ""}, {"reason": ""}, {"unexpected": "value"},
    {"action": "correct", "correction": {"arm": "left"}},
])
def test_worker_rejects_bad_matching_decision(tmp_path, answer):
    request, paths = fixture(tmp_path)
    identity = request_identity(request, paths)
    opener = Opener([decision(identity, **answer)])
    with pytest.raises(MailboxError):
        WorkerAossTeacher(config(), opener=opener).choose(request, paths, tmp_path / "worker")


def test_worker_fails_on_teacher_failure_envelope(tmp_path):
    request, paths = fixture(tmp_path)
    opener = Opener([{"request_id": request_identity(request, paths), "error": "teacher_failed"}])
    with pytest.raises(MailboxError, match="Local teacher failed"):
        WorkerAossTeacher(config(), opener=opener).choose(request, paths, tmp_path / "worker")


def test_worker_network_exception_is_sanitized(tmp_path):
    request, paths = fixture(tmp_path)
    with pytest.raises(MailboxError) as caught:
        WorkerAossTeacher(config(), opener=Opener([], fail_put=True)).choose(request, paths, tmp_path / "worker")
    assert "SECRET" not in str(caught.value)
    assert all("SECRET" not in file.read_text() for file in (tmp_path / "worker").iterdir())


@pytest.mark.parametrize("values", [
    {"timeout_seconds": 181}, {"timeout_seconds": True}, {"timeout_seconds": float("inf")},
    {"request_put": "http://aoss.cn-sh-01b.sensecoreapi-oss.cn/request?Signature=x"},
    {"request_put": "https://untrusted.example/request?Signature=x"},
    {"request_put": "https://aoss.cn-sh-01b.sensecoreapi-oss.cn/otherbucket/x?Signature=x"},
    {"request_put": "https://aoss.cn-sh-01b.sensecoreapi-oss.cn/260010165/otherprefix/x?Signature=x"},
    {"request_put": config()["response_get"]}, {"secret_key": "never accepted"},
])
def test_invalid_worker_configuration_rejected(values):
    with pytest.raises(ValueError):
        WorkerAossTeacher(config(**values))


def test_relay_preserves_incoming_and_executes_hash_once_across_restart(tmp_path):
    request, paths = fixture(tmp_path, 5)
    s3, teacher = S3(), Teacher()
    instance = relay(tmp_path, s3, teacher)
    data = payload(request, paths)
    s3.objects[instance.request_key] = json.dumps(data).encode()
    assert instance.process_once() is True
    assert instance.process_once() is False
    assert len(teacher.calls) == 1 and teacher.calls[0][1] == 5
    assert teacher.calls[0][2] == 10
    assert teacher.timeout_seconds == 120
    assert len(list((instance.output / "incoming").glob("*/payload.json"))) == 1
    assert len(list((instance.output / "incoming").glob("*/image-*.png"))) == 5
    restarted = relay(tmp_path, s3, teacher)
    assert restarted.decisions == 1
    assert restarted.process_once() is False
    assert len(teacher.calls) == 1
    assert json.loads(s3.objects[instance.response_key]) == decision(data["request_id"])


def test_relay_failure_is_terminal_and_never_retried(tmp_path):
    request, paths = fixture(tmp_path)
    s3, teacher = S3(), Teacher(fail=True)
    instance = relay(tmp_path, s3, teacher)
    data = payload(request, paths)
    s3.objects[instance.request_key] = json.dumps(data).encode()
    assert instance.process_once() is True
    assert json.loads(s3.objects[instance.response_key]) == {"request_id": data["request_id"], "error": "teacher_failed"}
    assert relay(tmp_path, s3, teacher).process_once() is False
    assert len(teacher.calls) == 1
    assert "SECRET" not in (instance.output / "reservations" / data["request_id"] / "failure.json").read_text()


def test_relay_crash_reservation_prevents_model_reexecution(tmp_path):
    request, paths = fixture(tmp_path)
    s3, teacher = S3(), Teacher()
    instance = relay(tmp_path, s3, teacher)
    data = payload(request, paths)
    (instance.output / "reservations" / data["request_id"]).mkdir()
    s3.objects[instance.request_key] = json.dumps(data).encode()
    restarted = relay(tmp_path, s3, teacher)
    assert restarted.process_once() is False
    assert len(teacher.calls) == 0
    assert json.loads(s3.objects[instance.response_key])["error"] == "reserved_request_not_retried"


@pytest.mark.parametrize("change", ["wrong_id", "bad_image", "wrong_instance", "too_many_images", "oversized"])
def test_relay_rejects_invalid_snapshots_without_inference(tmp_path, change):
    request, paths = fixture(tmp_path)
    s3, teacher = S3(), Teacher()
    instance = relay(tmp_path, s3, teacher)
    data = payload(request, paths)
    if change == "wrong_id":
        data["request_id"] = "f"*64
    elif change == "bad_image":
        data["images"][0] = "!badbase64!"
    elif change == "wrong_instance":
        data["request"]["instance_id"] = 301
    elif change == "too_many_images":
        data["images"] *= 2
    s3.objects[instance.request_key] = b"x"*(MAX_BODY+1) if change == "oversized" else json.dumps(data).encode()
    assert instance.process_once() is False
    assert instance.process_once() is False
    assert len(teacher.calls) == 0
    assert len(list((instance.output / "incoming").glob("*/failure.json"))) == 1


def test_relay_uses_finite_time_and_reports_health(tmp_path, capsys):
    clock = Clock()
    instance = relay(tmp_path, clock=clock, max_seconds=2)
    result = instance.run()
    assert clock.now == 2
    assert result["status"] == "completed" and result["ready"] is False
    assert json.loads(capsys.readouterr().out)["ready"] is True
    assert json.loads(instance.client.objects[instance.health_key])["status"] == "completed"


def test_relay_decision_budget_stops_before_second_inference(tmp_path):
    request, paths = fixture(tmp_path)
    s3, teacher = S3(), Teacher()
    instance = relay(tmp_path, s3, teacher, max_decisions=1)
    s3.objects[instance.request_key] = json.dumps(payload(request, paths)).encode()
    result = instance.run()
    assert len(teacher.calls) == result["decisions"] == 1
    request["step"] = 1
    s3.objects[instance.request_key] = json.dumps(payload(request, paths)).encode()
    assert instance.process_once() is False
    assert len(teacher.calls) == 1


def test_relay_rejects_out_of_scope_keys(tmp_path):
    with pytest.raises(ValueError):
        AossMailboxRelay(S3(), Teacher(), request_key="unrelated/request", response_key=PREFIX+"response",
                         evidence_root=tmp_path, max_decisions=1, max_seconds=1)

import json
import io
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
import pytest

from codex_collection_teacher import request_identity
from codex_teacher_bridge import RemoteTeacher, make_server


class Teacher:
    timeout_seconds = 120
    def __init__(self):
        self.calls = []
    def choose(self, request, images, output):
        self.calls.append(self.timeout_seconds)
        return dict(request_id=request_identity(request, images), action="continue", subtask="grasp the radio", reason="fixture")


@pytest.fixture
def bridge(tmp_path):
    token = tmp_path / "token"
    token.write_text("x" * 40)
    teacher = Teacher()
    server = make_server(0, token.read_text(), teacher, tmp_path / "evidence", 1, max_seconds=30)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server, teacher, RemoteTeacher(f"http://127.0.0.1:{server.server_port}", token)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def inputs(tmp_path):
    images = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4), (i, 0, 0)).save(path)
        images.append(path)
    request = dict(task="turning_on_radio", instance_id=311, attempt_id="test", step=0, previous_subtask=None)
    return request, images


def test_http_roundtrip_binds_images_and_clamps_remaining_time(tmp_path, bridge):
    server, teacher, client = bridge
    request, images = inputs(tmp_path)
    decision = client.choose(request, images, tmp_path / "client")
    assert decision["request_id"] == request_identity(request, images)
    assert server.decisions == 1 and 0 < teacher.calls[0] <= 30
    assert teacher.timeout_seconds == 120
    with pytest.raises(HTTPError) as error:
        client.choose(request, images, tmp_path / "repeat")
    assert error.value.code == 429 and len(teacher.calls) == 1


@pytest.mark.parametrize("native", [True, False])
def test_empty_continue_is_accepted_only_for_request_bound_native_mode(tmp_path, monkeypatch, native):
    import codex_teacher_bridge as module
    token = tmp_path / "token"; token.write_text("x" * 40)
    client = RemoteTeacher("http://127.0.0.1:1", token)
    request, images = inputs(tmp_path)
    if native:
        request.update(collection_mode="native_with_corrections", step=256,
            current_native_subtask="reach the radio", current_native_subtask_source="generated")
    response = dict(request_id=request_identity(request, images), action="continue", subtask="",
                    correction=None, reason="native progress visible")
    class Opener:
        def open(self, req, timeout):
            assert json.loads(req.data)["request"] == request
            return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    if native:
        assert client.choose(request, images, tmp_path / "client")["subtask"] == ""
    else:
        with pytest.raises(ValueError): client.choose(request, images, tmp_path / "client")


def test_wrong_token_cannot_consume_budget(tmp_path, bridge):
    server, teacher, client = bridge
    client.token = "wrong"
    with pytest.raises(HTTPError) as error:
        client.choose(*inputs(tmp_path), tmp_path / "response")
    assert error.value.code == 401 and server.decisions == 0 and teacher.calls == []


def test_expired_server_does_not_call_codex(tmp_path, bridge):
    server, teacher, client = bridge
    server.deadline = time.monotonic() - 1
    with pytest.raises(HTTPError) as error:
        client.choose(*inputs(tmp_path), tmp_path / "response")
    assert error.value.code == 429 and teacher.calls == []


def test_malformed_snapshot_rejected_before_teacher(tmp_path, bridge):
    server, teacher, client = bridge
    req = Request(client.url, data=b'{"request":{},"images":[]}',
                  headers={"Authorization": "Bearer " + client.token, "Content-Type": "application/json"})
    with pytest.raises(HTTPError) as error:
        urlopen(req, timeout=2)
    assert error.value.code == 400 and teacher.calls == []


@pytest.mark.parametrize("url", ["http://example.com", "https://127.0.0.1", "http://user:secret@127.0.0.1", "http://127.0.0.1?token=secret"])
def test_remote_public_endpoint_not_allowed(tmp_path, url):
    with pytest.raises(ValueError):
        RemoteTeacher(url, tmp_path / "unused")

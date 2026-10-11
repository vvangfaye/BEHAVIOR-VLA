"""Real local crypto tests using dummy documents, never real account files."""
import json
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_auth_handoff import HandoffError, encrypt, install, keygen


@pytest.fixture
def recipient(tmp_path):
    directory = tmp_path / "private-cluster-home"
    public = keygen(directory)
    return directory, public


def dummy_auth(tmp_path, payload=None):
    path = tmp_path / "explicit-dummy-account.json"
    path.write_text(json.dumps(payload or {"auth_mode": "chatgpt", "OPENAI_API_KEY": None,
        "tokens": {"access_token": "fixture-access-not-real", "refresh_token": "fixture-refresh-not-real",
                   "id_token": "fixture-id-not-real", "account_id": "fixture-account-not-real"}}))
    path.chmod(0o600)
    return path


def test_actual_openssl_roundtrip_does_not_publish_credentials(tmp_path, recipient):
    directory, public = recipient
    source = dummy_auth(tmp_path)
    ciphertext = tmp_path / "ciphertext.cms.der"
    encrypted = encrypt(source, Path(public["public_certificate_path"]), ciphertext, public["certificate_sha256"])
    receipt = install(directory, ciphertext, public["certificate_sha256"])
    assert (directory / "auth.json").read_bytes() == source.read_bytes()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "recipient-key.pem").stat().st_mode) == 0o600
    assert stat.S_IMODE((directory / "auth.json").stat().st_mode) == 0o600
    assert stat.S_IMODE(ciphertext.stat().st_mode) == 0o600
    assert receipt["readback_verified"] and receipt["live_login_verified"] is False
    assert encrypted["ciphertext_sha256"] == receipt["ciphertext_sha256"]
    assert receipt["status"] == "auth_installed"
    assert b"fixture-access-not-real" not in ciphertext.read_bytes()
    for report in (public, encrypted, receipt):
        assert "fixture-access" not in json.dumps(report)
        assert "fixture-refresh" not in json.dumps(report)
    assert keygen(directory, public["certificate_sha256"])["reused"]
    with pytest.raises(HandoffError, match="already exists"):
        install(directory, ciphertext, public["certificate_sha256"])
    with pytest.raises(HandoffError, match="already exists"):
        encrypt(source, Path(public["public_certificate_path"]), ciphertext, public["certificate_sha256"])


def test_recipient_reuse_requires_matching_fingerprint(tmp_path, recipient):
    directory, public = recipient
    with pytest.raises(HandoffError, match="expected fingerprint"):
        keygen(directory)
    with pytest.raises(HandoffError, match="fingerprint mismatch"):
        keygen(directory, "0" * 64)
    with pytest.raises(HandoffError, match="fingerprint mismatch"):
        encrypt(dummy_auth(tmp_path), Path(public["public_certificate_path"]), tmp_path / "bad.cms", "0" * 64)
    assert not (tmp_path / "bad.cms").exists()


@pytest.mark.parametrize("payload", [
    {"OPENAI_API_KEY": "fixture-key"},
    {"auth_mode": "api_key", "tokens": {"access_token": "dummy", "refresh_token": "dummy", "id_token": "dummy"}},
    {"tokens": {"access_token": "dummy", "refresh_token": "dummy"}},
])
def test_non_chatgpt_or_incomplete_documents_never_install(tmp_path, recipient, payload):
    directory, public = recipient
    cipher = tmp_path / "bad.cms"
    encrypt(dummy_auth(tmp_path, payload), Path(public["public_certificate_path"]), cipher, public["certificate_sha256"])
    with pytest.raises(HandoffError, match="ChatGPT authentication"):
        install(directory, cipher, public["certificate_sha256"])
    assert not (directory / "auth.json").exists()


def test_broader_directory_permissions_fail_before_key_creation(tmp_path):
    directory = tmp_path / "shared-home"
    directory.mkdir(mode=0o755)
    with pytest.raises(HandoffError, match="0700"):
        keygen(directory, "0" * 64)
    assert not list(directory.iterdir())


def test_auth_symlink_never_overwritten(tmp_path, recipient):
    directory, public = recipient
    source = dummy_auth(tmp_path)
    cipher = tmp_path / "secret.cms"
    encrypt(source, Path(public["public_certificate_path"]), cipher, public["certificate_sha256"])
    (directory / "auth.json").symlink_to(tmp_path / "missing-target")
    with pytest.raises(HandoffError, match="already exists"):
        install(directory, cipher, public["certificate_sha256"])
    assert not (tmp_path / "missing-target").exists()


def test_private_key_permissions_checked_on_reuse(tmp_path, recipient):
    directory, public = recipient
    (directory / "recipient-key.pem").chmod(0o644)
    with pytest.raises(HandoffError, match="permissions"):
        keygen(directory, public["certificate_sha256"])

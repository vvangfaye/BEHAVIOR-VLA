"""Install an encrypted fresh login and replay one teacher request without a GPU."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


def run(config):
    os.umask(0o077)
    root = Path(config["run_root"])
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    report = dict(kind="codex_subtask_cpu_teacher_validation_v1", state="starting",
                  selected_auth=config.get("selected_auth", "auth_weizhi.json"), model_calls=0, simulator_actions=0,
                  new_private_home=config["new_private_home"])
    sys.path[:0] = [str(Path(config["code"]) / name) for name in ("sco", "scripts")]
    from codex_cluster_proxy import CodexHomeLease, MihomoProxy
    from codex_subtask_teacher import CodexSubtaskTeacher
    cipher = root / "encrypted_login.cms"
    stage = "ciphertext"
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(config["ciphertext_url"], timeout=30) as response:
            raw = response.read(2 * 1024**2 + 1)
        if hashlib.sha256(raw).hexdigest() != config["ciphertext_sha256"]:
            raise ValueError("Encrypted login identity differs")
        cipher.write_bytes(raw)
        cipher.chmod(0o600)
        old = Path(config["old_private_home"])
        stage = "install_fresh_login"
        with CodexHomeLease(old):
            certificate = subprocess.run(["openssl", "x509", "-in", str(old / "recipient-cert.pem"),
                "-outform", "DER"], capture_output=True, check=True, timeout=15).stdout
            if hashlib.sha256(certificate).hexdigest() != config["certificate_sha256"]:
                raise ValueError("Private recipient certificate differs")
            decrypted = subprocess.run(["openssl", "cms", "-decrypt", "-binary", "-inform", "DER",
                "-in", str(cipher), "-recip", str(old / "recipient-cert.pem"),
                "-inkey", str(old / "recipient-key.pem")], capture_output=True, check=True, timeout=15).stdout
        if not 0 < len(decrypted) <= 1024**2:
            raise ValueError("Decrypted login exceeds finite size")
        value = json.loads(decrypted)
        tokens = value.get("tokens", {})
        if (value.get("auth_mode", "chatgpt") != "chatgpt" or value.get("OPENAI_API_KEY")
                or any(not isinstance(tokens.get(k), str) or not tokens[k] for k in (
                    "access_token", "refresh_token", "id_token"))
                or hashlib.sha256(str(tokens.get("account_id")).encode()).hexdigest()[:12]
                    != config["account_fingerprint"]):
            raise ValueError("Selected ChatGPT login differs")
        home = Path(config["new_private_home"])
        home.mkdir(mode=0o700, exist_ok=False)
        with os.fdopen(os.open(home / "auth.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(decrypted)
        if (home / "auth.json").read_bytes() != decrypted:
            raise ValueError("Private authentication installation readback differs")
        (home / "config.toml").write_text('cli_auth_credentials_store = "file"\n')
        (home / "config.toml").chmod(0o600)
        cipher.unlink()
        report.update(authentication_installed=True, credential_permissions="0600", private_home_permissions="0700")
        stage = "real_teacher_request"
        source = Path(config["source_request_dir"])
        frozen = json.loads((source / "request.json").read_text())
        images = []
        for row in frozen["images"]:
            path = source / row["file"]
            if path.parent != source or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Frozen teacher observation changed")
            images.append(path)
        with CodexHomeLease(home), MihomoProxy(config_dir=Path(config["mihomo_config_dir"]),
                run_dir=root / "teacher-network", mihomo_binary="/usr/local/bin/mihomo") as proxy:
            prior = dict(os.environ)
            try:
                os.environ.clear()
                os.environ.update(proxy.environment(prior, codex_home=home))
                teacher = CodexSubtaskTeacher(codex_binary="/usr/local/bin/codex", timeout_seconds=120,
                                             **config.get('teacher_configuration', {}))
                report["model_calls"] = 1
                decision = teacher.choose(frozen["request"], images, root / "probe")
            finally:
                os.environ.clear()
                os.environ.update(prior)
        receipt_path = root / "probe" / decision["request_id"] / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["status"] != "validated" or receipt["tool_event_validation"]["status"] != "no_tool_events_observed":
            raise ValueError("CPU teacher request failed its normal audit")
        report.update(state="validated", model_receipt=receipt, decision=decision,
                      input_kind="frozen_failed_attempt_observation_for_connectivity_only",
                      ready_for_new_gpu_attempt=True)
    except BaseException as error:
        report.update(state="failed", stage=stage, error_type=type(error).__name__, ready_for_new_gpu_attempt=False)
        receipts = list((root / "probe").glob("*/receipt.json"))
        if receipts:
            report["model_receipt"] = json.loads(receipts[-1].read_text())
    finally:
        if cipher.exists():
            cipher.unlink()
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

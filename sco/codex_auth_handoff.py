#!/usr/bin/env python3
"""Encrypt an explicitly selected Codex login for one private cluster home.

Only the public certificate and token-free status may be published. This helper
does not upload, download, select an account, or delete cloud objects. The caller
removes the transported ciphertext after confirming a successful installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile


class HandoffError(RuntimeError):
    pass


def _openssl(*args: str, timeout: int = 30) -> bytes:
    try:
        result = subprocess.run(["openssl", *map(str, args)], capture_output=True,
                                check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise HandoffError("OpenSSL operation could not complete") from None
    if result.returncode:
        # OpenSSL output can include input paths or decrypted material on errors.
        raise HandoffError("OpenSSL operation failed; private output withheld")
    return result.stdout


def _regular_file(path: Path, *, private: bool = False) -> None:
    try:
        info = path.lstat()
    except OSError:
        raise HandoffError("Required file is unavailable") from None
    if not stat.S_ISREG(info.st_mode) or (private and stat.S_IMODE(info.st_mode) != 0o600):
        raise HandoffError("Required file has unsafe type or permissions")


def _private_directory(directory: Path, *, create: bool = False) -> bool:
    created = False
    if create:
        try:
            directory.mkdir(mode=0o700)
            created = True
        except FileExistsError:
            pass
    try:
        info = directory.lstat()
    except OSError:
        raise HandoffError("Private directory is unavailable") from None
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise HandoffError("Private directory must be a real directory with mode 0700")
    return created


def certificate_fingerprint(certificate: Path) -> str:
    _regular_file(certificate)
    return hashlib.sha256(_openssl("x509", "-in", str(certificate), "-outform", "DER")).hexdigest()


def _require_fingerprint(certificate: Path, expected: str) -> str:
    actual = certificate_fingerprint(certificate)
    if not isinstance(expected, str) or expected.lower() != actual:
        raise HandoffError("Recipient certificate fingerprint mismatch")
    return actual


def _matching_key(directory: Path, expected: str) -> str:
    key, certificate = directory / "recipient-key.pem", directory / "recipient-cert.pem"
    _regular_file(key, private=True)
    fingerprint = _require_fingerprint(certificate, expected)
    key_public = _openssl("pkey", "-in", str(key), "-pubout")
    cert_public = _openssl("x509", "-in", str(certificate), "-pubkey", "-noout")
    if key_public.strip() != cert_public.strip():
        raise HandoffError("Recipient key does not match the certificate")
    return fingerprint


def keygen(directory: Path, expected_fingerprint: str | None = None) -> dict:
    directory = Path(directory).absolute()
    created = _private_directory(directory, create=True)
    key, certificate = directory / "recipient-key.pem", directory / "recipient-cert.pem"
    if created:
        # A fresh, private directory owns all intermediate files. Partial failures
        # stay visible and are not silently replaced with a different recipient.
        key.touch(mode=0o600, exist_ok=False)
        _openssl("req", "-x509", "-newkey", "rsa:3072", "-nodes", "-sha256",
                 "-keyout", str(key), "-out", str(certificate), "-days", "2",
                 "-subj", "/CN=Codex private cluster authentication handoff")
        key.chmod(0o600)
        certificate.chmod(0o644)
        fingerprint = certificate_fingerprint(certificate)
        if expected_fingerprint is not None and expected_fingerprint.lower() != fingerprint:
            raise HandoffError("A new recipient cannot match a preselected fingerprint")
    else:
        if expected_fingerprint is None:
            raise HandoffError("Reusing a private directory requires its expected fingerprint")
        fingerprint = _matching_key(directory, expected_fingerprint)
    return {"status": "recipient_ready", "reused": not created,
            "public_certificate_path": str(certificate), "certificate_sha256": fingerprint,
            "public_certificate_pem": certificate.read_text(encoding="ascii")}


def _exclusive_write(path: Path, content: bytes) -> None:
    # Hard-link a complete private file into place: readers never see a partial
    # auth file, and an existing target or symlink cannot be overwritten.
    descriptor, temporary = tempfile.mkstemp(prefix=".handoff-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            raise HandoffError("Destination already exists; refusing to replace it") from None
    finally:
        temporary_path.unlink(missing_ok=True)


def encrypt(auth_file: Path, certificate: Path, output: Path, expected_fingerprint: str) -> dict:
    auth_file, certificate, output = map(Path, (auth_file, certificate, output))
    _regular_file(auth_file)
    if not 0 < auth_file.stat().st_size <= 1048576:
        raise HandoffError("Selected authentication file exceeds the size bound")
    fingerprint = _require_fingerprint(certificate, expected_fingerprint)
    if output.exists() or output.is_symlink():
        raise HandoffError("Ciphertext destination already exists")
    ciphertext = _openssl("cms", "-encrypt", "-binary", "-aes256", "-in", str(auth_file),
                          "-outform", "DER", str(certificate))
    _exclusive_write(output, ciphertext)
    return {"status": "encrypted", "ciphertext_path": str(output.absolute()),
            "ciphertext_bytes": len(ciphertext), "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
            "certificate_sha256": fingerprint}


def _validate_auth(raw: bytes) -> None:
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate JSON key")
            result[name] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(value, dict) or value.get("auth_mode", "chatgpt") != "chatgpt":
            raise ValueError("Not a ChatGPT login")
        if value.get("OPENAI_API_KEY"):
            raise ValueError("API-key login is not permitted")
        tokens = value["tokens"]
        if not isinstance(tokens, dict) or any(not isinstance(tokens.get(key), str) or not tokens[key].strip()
                                              for key in ("access_token", "refresh_token", "id_token")):
            raise ValueError("Incomplete ChatGPT tokens")
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise HandoffError("Decrypted file is not a complete ChatGPT authentication document") from None


def install(directory: Path, ciphertext: Path, expected_fingerprint: str) -> dict:
    directory, ciphertext = Path(directory).absolute(), Path(ciphertext)
    _private_directory(directory)
    fingerprint = _matching_key(directory, expected_fingerprint)
    _regular_file(ciphertext)
    if not 0 < ciphertext.stat().st_size <= 2097152:
        raise HandoffError("Ciphertext exceeds the size bound")
    destination = directory / "auth.json"
    if destination.exists() or destination.is_symlink():
        raise HandoffError("Authentication destination already exists")
    raw = _openssl("cms", "-decrypt", "-binary", "-inform", "DER", "-in", str(ciphertext),
                   "-recip", str(directory / "recipient-cert.pem"),
                   "-inkey", str(directory / "recipient-key.pem"))
    if not 0 < len(raw) <= 1048576:
        raise HandoffError("Decrypted file exceeds the size bound")
    _validate_auth(raw)
    _exclusive_write(destination, raw)
    _regular_file(destination, private=True)
    if destination.read_bytes() != raw:
        raise HandoffError("Authentication file readback differs from the decrypted content")
    return {"status": "auth_installed", "auth_path": str(destination), "auth_mode": "chatgpt",
            "auth_permissions": "0600", "private_directory_permissions": "0700",
            "certificate_sha256": fingerprint, "readback_verified": True,
            "ciphertext_sha256": hashlib.sha256(ciphertext.read_bytes()).hexdigest(),
            "live_login_verified": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    recipient = commands.add_parser("keygen")
    recipient.add_argument("--directory", type=Path, required=True)
    recipient.add_argument("--expected-fingerprint")
    sender = commands.add_parser("encrypt")
    sender.add_argument("--auth-file", type=Path, required=True)
    sender.add_argument("--certificate", type=Path, required=True)
    sender.add_argument("--output", type=Path, required=True)
    sender.add_argument("--expected-fingerprint", required=True)
    receiver = commands.add_parser("install")
    receiver.add_argument("--directory", type=Path, required=True)
    receiver.add_argument("--ciphertext", type=Path, required=True)
    receiver.add_argument("--expected-fingerprint", required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "keygen":
            result = keygen(args.directory, args.expected_fingerprint)
        elif args.operation == "encrypt":
            result = encrypt(args.auth_file, args.certificate, args.output, args.expected_fingerprint)
        else:
            result = install(args.directory, args.ciphertext, args.expected_fingerprint)
    except (HandoffError, OSError):
        print(json.dumps({"status": "failed", "operation": args.operation,
                          "error": "Authentication handoff failed; private details withheld"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

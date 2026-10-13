"""Private, finite mihomo runtime for the cluster's existing Codex login.

The source proxy configuration and Codex credentials are never rewritten or
logged. Each worker clones the proxy assets (including stored selections), owns
one loopback listener, and removes its private copy on exit. The Codex home
lease coordinates this experiment's workers; unrelated clients must not use the
same home concurrently because they do not necessarily honor this lease.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import copy
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time
from typing import Mapping


class ProxyRuntimeError(RuntimeError):
    """A deliberately redacted failure; never include config or process output."""


def _positive_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 120:
        raise ValueError("Proxy timeout must be positive and at most 120 seconds")
    return float(value)


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700)
    path.chmod(0o700)


def _contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _clone_assets(source: Path, target: Path, *, max_bytes: int = 1024 ** 3,
                  max_files: int = 10000) -> dict:
    """Copy regular assets, rejecting external links, special files, and races."""
    totals = {"files": 0, "bytes": 0}
    for current, directories, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        relative = current_path.relative_to(source)
        destination = target / relative
        for name in directories:
            entry = current_path / name
            if entry.is_symlink():
                raise ProxyRuntimeError("Proxy asset directory links require explicit staging")
            _private_directory(destination / name)
        for name in files:
            entry = current_path / name
            resolved = entry.resolve(strict=True)
            if not _contained(resolved, source):
                raise ProxyRuntimeError("Proxy assets may not link outside their configuration directory")
            before = resolved.stat()
            if not stat.S_ISREG(before.st_mode):
                raise ProxyRuntimeError("Proxy assets must be ordinary files")
            totals["files"] += 1
            totals["bytes"] += before.st_size
            if totals["files"] > max_files or totals["bytes"] > max_bytes:
                raise ProxyRuntimeError("Proxy asset clone exceeds the bounded size budget")
            output = destination / name
            with resolved.open("rb") as reader, output.open("xb") as writer:
                output.chmod(0o600)
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            after = resolved.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ProxyRuntimeError("Proxy assets changed during the private snapshot")
    return totals


def _asset_path(value: str, source: Path, target: Path) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProxyRuntimeError("Proxy provider has an invalid local asset path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = source / candidate
    resolved = candidate.resolve()
    if not _contained(resolved, source):
        raise ProxyRuntimeError("Proxy provider paths must remain in the cloned configuration directory")
    return str(target / resolved.relative_to(source))


def private_configuration(configuration: dict, *, source: Path, target: Path, port: int) -> dict:
    """Preserve outbound routing and stored selectors; remove shared listeners."""
    if not isinstance(configuration, dict) or type(port) is not int or not 1024 <= port <= 65535:
        raise ProxyRuntimeError("Proxy configuration or loopback port is invalid")
    result = copy.deepcopy(configuration)
    mode = result.get("mode", "rule")
    if not isinstance(mode, str) or mode.casefold() not in {"rule", "global", "direct"}:
        raise ProxyRuntimeError("Proxy configuration has an unsupported routing mode")
    # User profiles may spell this enum as Rule/Global/Direct. Normalize only
    # its case; retain the same routing choice and reject other values.
    result["mode"] = mode.casefold()
    for family in ("proxy-providers", "rule-providers"):
        providers = result.get(family, {})
        if not isinstance(providers, dict):
            raise ProxyRuntimeError("Proxy provider configuration must be a mapping")
        for provider in providers.values():
            if not isinstance(provider, dict):
                raise ProxyRuntimeError("Proxy provider configuration must be a mapping")
            if "path" in provider:
                provider["path"] = _asset_path(provider["path"], source, target)
    # The mixed listener serves HTTP CONNECT and SOCKS on this worker only.
    result.update({"mixed-port": port, "port": 0, "socks-port": 0,
                   "redir-port": 0, "tproxy-port": 0, "allow-lan": False,
                   "bind-address": "127.0.0.1", "authentication": [],
                   "skip-auth-prefixes": ["127.0.0.1/32"], "log-level": "silent",
                   "listeners": [], "tunnels": [], "tun": {"enable": False},
                   "ss-config": "", "vmess-config": "", "tuic-server": {"enable": False},
                   "iptables": {"enable": False}})
    # A control API is unnecessary for this run. Saved choices are loaded from
    # the private cache.db with the unchanged profile.store-selected setting.
    for key in list(result):
        if key.startswith("external-controller") or key.startswith("external-ui"):
            result.pop(key)
    for key in ("external-doh-server", "secret", "tls", "interface-name", "routing-mark"):
        result.pop(key, None)
    dns = result.get("dns")
    if dns is not None:
        if not isinstance(dns, dict):
            raise ProxyRuntimeError("Proxy DNS configuration must be a mapping")
        dns.pop("listen", None)  # Preserve internal DNS resolution, no DNS server.
    ntp = result.get("ntp")
    if ntp is not None:
        if not isinstance(ntp, dict):
            raise ProxyRuntimeError("Proxy NTP configuration must be a mapping")
        ntp["enable"] = False
        ntp["write-to-system"] = False
    return result


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


def child_environment(inherited: Mapping[str, str], *, port: int, codex_home: Path | None = None) -> dict:
    """Proxy only this child tree; never modify the caller's process environment."""
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Proxy port must be an unprivileged TCP port")
    env = {key: value for key, value in inherited.items()
           if key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                                  "OPENAI_API_KEY", "CODEX_API_KEY"}}
    proxy = f"http://127.0.0.1:{port}"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[name] = proxy
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    if codex_home is not None:
        env["CODEX_HOME"] = str(codex_home)
    return env


class CodexHomeLease(AbstractContextManager):
    """An advisory single-writer lease, without opening or copying auth.json."""

    def __init__(self, home: Path, *, wait_timeout: float = 0):
        self.home = Path(home).resolve(strict=True)
        if not self.home.is_dir():
            raise ValueError("The existing Codex home must be a directory")
        if type(wait_timeout) not in (int, float) or not math.isfinite(wait_timeout) or not 0 <= wait_timeout <= 300:
            raise ValueError("Lease wait must be finite and between zero and 300 seconds")
        self.wait_timeout = float(wait_timeout)
        self.waited_seconds = 0.0
        self._fd = None

    def __enter__(self):
        if self._fd is not None:
            raise RuntimeError("Codex home lease is already active")
        try:
            descriptor = os.open(self.home / ".collection-worker.lock",
                                 os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            os.fchmod(descriptor, 0o600)
            try:
                started = time.monotonic()
                while True:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self.waited_seconds = time.monotonic() - started
                        break
                    except BlockingIOError:
                        remaining = self.wait_timeout - (time.monotonic() - started)
                        if remaining <= 0:
                            raise
                        time.sleep(min(0.1, remaining))
            except Exception:
                os.close(descriptor)
                raise
        except Exception:
            raise ProxyRuntimeError("The existing Codex home is unavailable for an exclusive worker lease") from None
        self._fd = descriptor
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        # Never unlink a lockfile: another process might already hold its inode.
        return False


class MihomoProxy(AbstractContextManager):
    """Start only this worker's mihomo process and clean it up within a bound."""

    def __init__(self, *, config_dir: Path, run_dir: Path,
                 mihomo_binary: str = "/usr/local/bin/mihomo", config_name: str = "config.yaml",
                 startup_timeout: float = 45, shutdown_timeout: float = 10,
                 port: int | None = None):
        self.source = Path(config_dir).resolve(strict=True)
        self.run_dir = Path(run_dir).resolve()
        self.private_dir = self.run_dir / "private-mihomo"
        if not self.source.is_dir() or _contained(self.run_dir, self.source) or _contained(self.source, self.run_dir):
            raise ValueError("Proxy source and private runtime must be separate directories")
        if not isinstance(config_name, str) or Path(config_name).name != config_name:
            raise ValueError("Proxy configuration name must be a filename")
        if not isinstance(mihomo_binary, str) or not mihomo_binary:
            raise ValueError("Mihomo binary must name an executable")
        self.config_name = config_name
        self.binary = mihomo_binary
        self.startup_timeout = _positive_timeout(startup_timeout)
        self.shutdown_timeout = _positive_timeout(shutdown_timeout)
        self.port = _unused_loopback_port() if port is None else port
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise ValueError("Proxy port must be an unprivileged TCP port")
        self.process = None
        self._owns_private = False
        self.receipt = {"kind": "codex_cluster_proxy_v1", "state": "not_started",
                        "loopback_only": True, "tun_enabled": False,
                        "source_configuration_modified": False,
                        "codex_credentials_copied": False}

    def __enter__(self):
        import yaml
        if self.process is not None or self._owns_private:
            raise RuntimeError("Proxy context cannot be entered twice")
        if self.private_dir.exists():
            raise ProxyRuntimeError("Private proxy output already exists; refusing overwrite")
        if not self.run_dir.exists():
            self.run_dir.mkdir(mode=0o700, parents=True)
        if self.run_dir.is_symlink() or self.run_dir.stat().st_mode & 0o077:
            raise ProxyRuntimeError("Private proxy runtime directory must have mode 0700")
        try:
            _private_directory(self.private_dir)
            self._owns_private = True
            counts = _clone_assets(self.source, self.private_dir)
            config_path = self.private_dir / self.config_name
            if not config_path.is_file() or config_path.stat().st_size > 16 * 1024 * 1024:
                raise ProxyRuntimeError("Proxy configuration is missing or exceeds the size budget")
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except Exception:
                raise ProxyRuntimeError("Proxy configuration is not valid YAML") from None
            config = private_configuration(config, source=self.source, target=self.private_dir, port=self.port)
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            config_path.chmod(0o600)
            self.receipt.update(asset_files=counts["files"], asset_bytes=counts["bytes"],
                                routing_mode=config.get("mode", "rule"),
                                selection_cache_copied=(self.private_dir / "cache.db").is_file(),
                                loopback_port=self.port)
            # The proxy itself must not inherit another proxy or broad SAFE_PATHS.
            env = {key: value for key, value in os.environ.items()
                   if key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                                          "OPENAI_API_KEY", "CODEX_API_KEY", "SAFE_PATHS"}}
            command = [self.binary, "-d", str(self.private_dir), "-f", str(config_path)]
            checked = subprocess.run(command + ["-t"], env=env, cwd=self.private_dir,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, timeout=self.startup_timeout, check=False)
            if checked.returncode:
                raise ProxyRuntimeError("Private mihomo configuration validation failed")
            # Refuse an already occupied port; never treat an unrelated proxy as
            # this worker's successful start.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.bind(("127.0.0.1", self.port))
            self.process = subprocess.Popen(command, env=env, cwd=self.private_dir,
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise ProxyRuntimeError("Private mihomo exited before becoming ready")
                try:
                    with socket.create_connection(("127.0.0.1", self.port), timeout=.2):
                        self.receipt["state"] = "ready"
                        return self
                except OSError:
                    threading.Event().wait(min(.1, max(0, deadline - time.monotonic())))
            raise ProxyRuntimeError("Private mihomo did not become ready before its deadline")
        except Exception as error:
            self.receipt.update(state="failed", error_type=type(error).__name__)
            self.close()
            if isinstance(error, ProxyRuntimeError):
                raise
            raise ProxyRuntimeError("Private mihomo setup failed; configuration and output were suppressed") from None

    def environment(self, inherited: Mapping[str, str] | None = None, *, codex_home: Path | None = None) -> dict:
        if self.process is None or self.process.poll() is not None or self.receipt["state"] != "ready":
            raise ProxyRuntimeError("Private mihomo is not running")
        return child_environment(os.environ if inherited is None else inherited,
                                 port=self.port, codex_home=codex_home)

    def close(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=self.shutdown_timeout)
            self.process = None
        if self._owns_private:
            if self.private_dir.exists():
                shutil.rmtree(self.private_dir)
            self._owns_private = False
        if self.receipt["state"] == "ready":
            self.receipt["state"] = "closed"

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

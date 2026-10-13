import copy
import json
import os
from pathlib import Path
import socket
import stat
import sys

import pytest
import yaml

from codex_cluster_proxy import (CodexHomeLease, MihomoProxy, ProxyRuntimeError,
                                 child_environment, private_configuration)


def config_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "providers").mkdir()
    (source / "providers" / "nodes.yaml").write_text("proxies: []\n")
    (source / "cache.db").write_bytes(b"fixture-stored-selector-choices")
    config = {"mode": "global", "mixed-port": 7890, "allow-lan": True,
              "bind-address": "*", "authentication": ["SECRET:credential"],
              "external-controller": "0.0.0.0:9090", "external-controller-unix": "/tmp/shared.sock",
              "external-ui-url": "https://example.invalid/SECRET", "secret": "SECRET-controller",
              "tun": {"enable": True, "auto-route": True}, "listeners": [{"port": 8000}],
              "ss-config": "SECRET-ss-inbound", "vmess-config": "SECRET-vmess-inbound",
              "tuic-server": {"enable": True, "listen": "0.0.0.0:10443"},
              "iptables": {"enable": True},
              "tunnels": ["tcp/udp,127.0.0.1:6553,1.1.1.1:53,proxy"],
              "dns": {"enable": True, "listen": "0.0.0.0:53", "nameserver": ["1.1.1.1"]},
              "profile": {"store-selected": True, "store-fake-ip": True},
              "proxies": [{"name": "chosen", "type": "ss", "password": "SECRET-proxy"}],
              "proxy-groups": [{"name": "group", "type": "select", "proxies": ["chosen", "DIRECT"]}],
              "proxy-providers": {"subscription": {"type": "http", "url": "https://example.invalid/SECRET",
                                                     "path": str(source / "providers" / "nodes.yaml")}},
              "rule-providers": {"rules": {"type": "file", "path": "providers/nodes.yaml"}},
              "rules": ["MATCH,group"]}
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    runtime = tmp_path / "worker"
    runtime.mkdir(mode=0o700)
    return source, runtime, config


def fake_mihomo(tmp_path, *, failure=False, no_listener=False):
    path = tmp_path / "mihomo-fixture"
    source = f"""#!{sys.executable}
import argparse
import signal
import socket
import threading
import yaml
p = argparse.ArgumentParser()
p.add_argument('-d')
p.add_argument('-f')
p.add_argument('-t', action='store_true')
a = p.parse_args()
config = yaml.safe_load(open(a.f))
if a.t:
    print('SECRET-validation-output')
    raise SystemExit({3 if failure else 0})
if {no_listener!r}:
    threading.Event().wait(60)
else:
    listener = socket.socket()
    listener.bind((config['bind-address'], config['mixed-port']))
    listener.listen(4)
    while True:
        conn, address = listener.accept()
        conn.close()
"""
    path.write_text(source)
    path.chmod(0o700)
    return str(path)


def test_outbound_routing_and_selector_policy_are_preserved(tmp_path):
    source, runtime, original = config_fixture(tmp_path)
    untouched = copy.deepcopy(original)
    result = private_configuration(original, source=source, target=runtime, port=18482)
    assert original == untouched
    for key in ("mode", "profile", "proxies", "proxy-groups", "rules"):
        assert result[key] == original[key]
    assert result["proxy-providers"]["subscription"]["path"] == str(runtime / "providers/nodes.yaml")
    assert result["rule-providers"]["rules"]["path"] == str(runtime / "providers/nodes.yaml")
    assert result["allow-lan"] is False
    assert result["bind-address"] == "127.0.0.1"
    assert result["tun"] == {"enable": False}
    assert result["tuic-server"] == result["iptables"] == {"enable": False}
    assert result["ss-config"] == result["vmess-config"] == ""
    assert result["listeners"] == result["tunnels"] == result["authentication"] == []
    assert not [key for key in result if key.startswith("external-")]
    assert "listen" not in result["dns"]
    assert result["dns"]["nameserver"] == ["1.1.1.1"]
    assert all(result[key] == 0 for key in ("port", "socks-port", "redir-port", "tproxy-port"))


@pytest.mark.parametrize("path", ["../external", "/tmp/outside-provider"])
def test_provider_cannot_escape_private_assets(tmp_path, path):
    source, runtime, config = config_fixture(tmp_path)
    config["rule-providers"]["rules"]["path"] = path
    with pytest.raises(ProxyRuntimeError, match="cloned"):
        private_configuration(config, source=source, target=runtime, port=18482)


@pytest.mark.parametrize("configured,expected", [("Rule", "rule"), ("Global", "global"),
                                               ("Direct", "direct"), ("RULE", "rule")])
def test_routing_mode_case_is_normalized_without_changing_its_choice(tmp_path, configured, expected):
    source, runtime, config = config_fixture(tmp_path)
    config["mode"] = configured
    result = private_configuration(config, source=source, target=runtime, port=18482)
    assert result["mode"] == expected
    assert config["mode"] == configured
    assert result["proxy-groups"] == config["proxy-groups"]
    assert result["rules"] == config["rules"]


@pytest.mark.parametrize("configured", ["unknown", "Rule ", "", None, False, [], {}])
def test_unknown_routing_modes_are_still_rejected(tmp_path, configured):
    source, runtime, config = config_fixture(tmp_path)
    config["mode"] = configured
    with pytest.raises(ProxyRuntimeError, match="unsupported routing mode"):
        private_configuration(config, source=source, target=runtime, port=18482)


def test_child_environment_does_not_bypass_proxy_or_expose_api_keys(tmp_path):
    inherited = {"HOME": "/original/home", "PATH": "/usr/bin", "https_proxy": "SECRET-old-proxy",
                 "NO_PROXY": "*", "OPENAI_API_KEY": "SECRET-key", "CODEX_API_KEY": "SECRET-key",
                 "CODEX_HOME": "/previous/home"}
    snapshot = dict(inherited)
    result = child_environment(inherited, port=18482, codex_home=tmp_path)
    assert inherited == snapshot
    assert result["HOME"] == "/original/home"
    assert result["CODEX_HOME"] == str(tmp_path)
    assert result["NO_PROXY"] == "localhost,127.0.0.1,::1"
    assert result["HTTPS_PROXY"] == result["https_proxy"] == "http://127.0.0.1:18482"
    assert "SECRET" not in json.dumps(result)


def test_codex_home_lease_serializes_workers_without_reading_credentials(tmp_path):
    home = tmp_path / "codex"
    home.mkdir()
    auth = home / "auth.json"
    auth.write_text("SECRET-fixture-auth")
    before = auth.stat()
    with CodexHomeLease(home):
        with pytest.raises(ProxyRuntimeError, match="exclusive"):
            with CodexHomeLease(home):
                pytest.fail("Second writer acquired the existing account home")
    with CodexHomeLease(home):
        pass
    assert auth.read_text() == "SECRET-fixture-auth"
    assert auth.stat().st_mtime_ns == before.st_mtime_ns
    assert stat.S_IMODE((home / ".collection-worker.lock").stat().st_mode) == 0o600


def test_existing_lock_symlink_is_rejected_without_following_it(tmp_path):
    external = tmp_path / "keep"
    external.write_text("unchanged")
    home = tmp_path / "codex"
    home.mkdir()
    (home / ".collection-worker.lock").symlink_to(external)
    with pytest.raises(ProxyRuntimeError):
        with CodexHomeLease(home):
            pass
    assert external.read_text() == "unchanged"


def test_real_child_process_is_private_and_removed_on_exit(tmp_path):
    source, runtime, _ = config_fixture(tmp_path)
    original = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    proxy = MihomoProxy(config_dir=source, run_dir=runtime, mihomo_binary=fake_mihomo(tmp_path))
    with proxy:
        process = proxy.process
        assert process.poll() is None
        assert proxy.environment({})["HTTP_PROXY"].startswith("http://127.0.0.1:")
        assert (proxy.private_dir / "cache.db").read_bytes() == b"fixture-stored-selector-choices"
        for path in proxy.private_dir.rglob("*"):
            assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
        assert "SECRET" not in json.dumps(proxy.receipt)
    assert process.poll() is not None
    assert not proxy.private_dir.exists()
    assert proxy.receipt["state"] == "closed"
    assert original == {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}


def test_failed_config_validation_has_no_secret_output_or_leftover_copy(tmp_path, capsys):
    source, runtime, _ = config_fixture(tmp_path)
    proxy = MihomoProxy(config_dir=source, run_dir=runtime, mihomo_binary=fake_mihomo(tmp_path, failure=True))
    with pytest.raises(ProxyRuntimeError, match="validation failed"):
        with proxy:
            pytest.fail("Invalid proxy was started")
    assert "SECRET" not in str(capsys.readouterr()) + json.dumps(proxy.receipt)
    assert not proxy.private_dir.exists()


def test_startup_deadline_stops_its_child_and_removes_private_credentials(tmp_path):
    source, runtime, _ = config_fixture(tmp_path)
    proxy = MihomoProxy(config_dir=source, run_dir=runtime, mihomo_binary=fake_mihomo(tmp_path, no_listener=True),
                        startup_timeout=1, shutdown_timeout=.3)
    with pytest.raises(ProxyRuntimeError, match="deadline"):
        with proxy:
            pytest.fail("A listener that never starts cannot pass readiness")
    assert proxy.process is None
    assert not proxy.private_dir.exists()


def test_external_asset_symlink_is_rejected_and_source_untouched(tmp_path):
    source, runtime, _ = config_fixture(tmp_path)
    external = tmp_path / "credential"
    external.write_text("SECRET-unrelated")
    (source / "outside").symlink_to(external)
    proxy = MihomoProxy(config_dir=source, run_dir=runtime)
    with pytest.raises(ProxyRuntimeError, match="outside"):
        with proxy:
            pass
    assert not proxy.private_dir.exists()
    assert external.read_text() == "SECRET-unrelated"


def test_preexisting_private_output_is_not_deleted(tmp_path):
    source, runtime, _ = config_fixture(tmp_path)
    private = runtime / "private-mihomo"
    private.mkdir()
    (private / "keep").write_text("previous evidence")
    with pytest.raises(ProxyRuntimeError, match="already exists"):
        with MihomoProxy(config_dir=source, run_dir=runtime):
            pass
    assert (private / "keep").read_text() == "previous evidence"


def test_existing_listener_cannot_be_mistaken_for_own_proxy(tmp_path):
    source, runtime, _ = config_fixture(tmp_path)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        proxy = MihomoProxy(config_dir=source, run_dir=runtime, port=occupied.getsockname()[1],
                            mihomo_binary=fake_mihomo(tmp_path))
        with pytest.raises(ProxyRuntimeError, match="setup failed"):
            with proxy:
                pass
        assert proxy.process is None
        assert not proxy.private_dir.exists()


def test_yaml_error_does_not_reveal_secret_document(tmp_path):
    source, runtime, _ = config_fixture(tmp_path)
    (source / "config.yaml").write_text("password: [SECRET-unclosed\n")
    with pytest.raises(ProxyRuntimeError) as caught:
        with MihomoProxy(config_dir=source, run_dir=runtime):
            pass
    assert "SECRET" not in str(caught.value)
    assert "valid YAML" in str(caught.value)


def test_waiting_lease_acquires_after_release_and_times_out(tmp_path):
    import threading
    home = tmp_path / 'shared'
    home.mkdir()
    first = CodexHomeLease(home).__enter__()
    with pytest.raises(ProxyRuntimeError):
        with CodexHomeLease(home, wait_timeout=.03):
            pytest.fail('Concurrent writer admitted')
    release = threading.Timer(.05, first.__exit__, args=(None, None, None))
    release.start()
    with CodexHomeLease(home, wait_timeout=1) as second:
        assert second.waited_seconds >= .04
    release.join()
    with CodexHomeLease(home):
        pass

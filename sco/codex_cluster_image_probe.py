"""Finite, credential-safe inspection of the user's existing cluster image."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

root = Path('/mnt/afs/260010165/container-env')
report = {'kind': 'codex_cluster_image_probe_v1', 'state': 'inspecting', 'files': [], 'binaries': {}}

def publish():
    data = json.dumps(report).encode()
    request = urllib.request.Request(os.environ['CODEX_PILOT_REPORT_PUT'], data=data, method='PUT')
    with urllib.request.urlopen(request, timeout=30):
        pass

publish()
for directory, children, files in os.walk(root):
    depth = len(Path(directory).relative_to(root).parts)
    children[:] = [name for name in children if name not in {'node_modules', 'cache', 'logs', 'sessions', '.git'}]
    if depth >= 2:
        children[:] = []
    for name in files:
        if name in {'config.yaml', 'config.yml', 'auth.json', 'config.toml', 'env.sh', 'bootstrap.sh'} or depth == 0:
            path = Path(directory)/name
            stat = path.stat()
            report['files'].append(dict(path=str(path), bytes=stat.st_size, mode=oct(stat.st_mode & 0o777)))
            # Never read authentication or proxy configuration contents into logs.
            if name in {'env.sh', 'bootstrap.sh'}:
                report.setdefault('launcher_sha256', {})[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
for name in ('codex', 'mihomo', 'node', 'npm', 'curl'):
    executable = shutil.which(name)
    entry = {'path': executable}
    if executable:
        args = [executable, '-v' if name == 'mihomo' else '--version']
        completed = subprocess.run(args, capture_output=True, text=True, timeout=15)
        entry.update(returncode=completed.returncode, version=completed.stdout.strip()[:150])
        if name == 'codex':
            env = {k:v for k,v in os.environ.items() if k not in {'OPENAI_API_KEY','CODEX_API_KEY'}}
            result = subprocess.run([executable, 'login', 'status'], env=env, capture_output=True, text=True, timeout=15)
            text = (result.stdout or '')+'\n'+(result.stderr or '')
            entry.update(chatgpt_login=result.returncode == 0 and 'Logged in using ChatGPT' in text,
                         login_returncode=result.returncode)
    report['binaries'][name] = entry
report['environment_paths'] = {key:os.environ[key] for key in ('CODEX_HOME','XDG_CONFIG_HOME')
    if key in os.environ and os.environ[key].startswith('/')}
report['default_auth_file_exists'] = Path.home().joinpath('.codex/auth.json').is_file()
report['default_proxy_config_exists'] = Path.home().joinpath('.config/mihomo/config.yaml').is_file()
report['state'] = 'inspected'
publish()
print(json.dumps(report))

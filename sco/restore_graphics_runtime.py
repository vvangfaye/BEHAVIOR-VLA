"""Restore Ubuntu GL loaders on AFS without allocating a GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request


def run(payload):
    runtime = Path(payload['runtime_root'])
    root = Path(payload['graphics_root'])
    assert (runtime / '_RUNTIME_READY').is_file()
    assert not root.exists(), 'Graphics restoration requires a fresh directory'
    root.mkdir(parents=True)
    log = (root / 'driver.log').open('x', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    status = {'state': 'running', 'gpu_count': 0, 'started_at': time.time(),
              'runtime_root': str(runtime), 'graphics_root': str(root)}

    def publish():
        (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        if payload.get('report_put'):
            report = dict(status)
            with (root / 'driver.log').open('rb') as source:
                source.seek(max(0, os.fstat(source.fileno()).st_size - 10000))
                report['log_tail'] = source.read().decode('utf-8', errors='replace')
            try:
                with urllib.request.urlopen(urllib.request.Request(payload['report_put'],
                        data=json.dumps(report).encode(), method='PUT'), timeout=30):
                    pass
            except Exception:
                print('graphics_report_upload_failed', flush=True)

    try:
        publish()
        os_release = Path('/etc/os-release').read_text()
        assert 'ID=ubuntu' in os_release and 'VERSION_ID="22.04"' in os_release
        packages = ['libgl1', 'libglx0', 'libglvnd0', 'libglu1-mesa', 'libopengl0']
        downloads = root / 'packages'
        downloads.mkdir()
        # APT verifies downloads against the image's trusted Ubuntu package indexes.
        subprocess.run(['apt-get', '-o', 'Acquire::Retries=2',
            '-o', 'Acquire::http::Timeout=30', 'download', *packages],
            cwd=downloads, check=True, timeout=240)
        metadata = []
        for archive in sorted(downloads.glob('*.deb')):
            name = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Package'], text=True).strip()
            version = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Version'], text=True).strip()
            assert name in packages
            metadata.append({'package': name, 'version': version, 'file': archive.name,
                'bytes': archive.stat().st_size, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
            unpack = subprocess.Popen(['dpkg-deb', '--fsys-tarfile', str(archive)], stdout=subprocess.PIPE)
            extract = subprocess.run(['tar', '--no-same-owner', '-xf', '-', '-C', str(root)],
                                     stdin=unpack.stdout, check=False, timeout=60)
            unpack.stdout.close()
            assert unpack.wait(timeout=30) == 0 and extract.returncode == 0
        assert {item['package'] for item in metadata} == set(packages)
        library_path = root / 'usr/lib/x86_64-linux-gnu'
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        environment.update(PYTHONNOUSERSITE='1', LD_LIBRARY_PATH=str(library_path) + ':' +
                           str(runtime / 'envs/b1k-eval/lib') + ':' + environment.get('LD_LIBRARY_PATH', ''))
        names = ['libGL.so.1', 'libGLX.so.0', 'libGLdispatch.so.0', 'libGLU.so.1', 'libOpenGL.so.0']
        subprocess.run([str(runtime / 'envs/b1k-eval/bin/python'), '-c',
            'import ctypes;[ctypes.CDLL(name) for name in ' + repr(names) + '];print("GL_loaders_passed")'],
            check=True, env=environment, timeout=90)
        neuray = runtime / ('envs/b1k-eval/lib/python3.11/site-packages/isaacsim/extscache/'
            'omni.iray.libs-0.0.0+69cbf6ad.lx64.r/bin/iray/libneuray.so')
        resolved = subprocess.check_output(['ldd', str(neuray)], env=environment, text=True, timeout=30)
        assert 'not found' not in resolved, resolved
        libraries = {name: hashlib.sha256((library_path / name).resolve().read_bytes()).hexdigest()
                     for name in names}
        ready = {'packages': metadata, 'libraries_sha256': libraries,
                 'library_path': str(library_path), 'neuray_dependencies_resolved': True,
                 'gpu_validation': False, 'completed_at': time.time()}
        (root / '_READY').write_text(json.dumps(ready, indent=2) + '\n')
        link = runtime / 'graphics-runtime'
        assert not link.exists() and not link.is_symlink(), 'Do not replace another graphics snapshot'
        link.symlink_to(root, target_is_directory=True)
        status.update(state='complete', ready=ready, completed_at=time.time())
        publish()
    except BaseException as error:
        traceback.print_exc()
        status.update(state='failed', error_type=type(error).__name__, updated_at=time.time())
        publish()
        raise

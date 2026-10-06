#!/usr/bin/env python3
"""OSS API preflight and receipts for submission hosts without an OSS mount."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import urlsplit


class PreflightError(RuntimeError):
    pass


def object_key(path, bucket, mount_root='/data/oss_bucket_0'):
    if path.startswith('oss://'):
        url = urlsplit(path)
        if url.netloc != bucket or url.query or url.fragment:
            raise PreflightError('OSS path must use the configured bucket without query/fragment')
        key = url.path.lstrip('/')
    elif path.startswith(mount_root.rstrip('/') + '/'):
        key = path[len(mount_root.rstrip('/')) + 1:]
    else:
        raise PreflightError('OSS preflight requires oss:// paths or paths under OSS_MOUNT_ROOT')
    if not key or '..' in PurePosixPath(key).parts or '\n' in key or '\r' in key:
        raise PreflightError('Invalid OSS object path')
    return key.rstrip('/')


class OSSClient:
    def __init__(self):
        try:
            import oss2
        except ImportError:
            raise PreflightError('Install nebula/requirements-submit.txt and set NEBULA_SUBMIT_PYTHON') from None
        self.oss2 = oss2
        self.bucket_name = os.environ['OSS_BUCKET'].rstrip('/')
        self.mount_root = os.environ.get('OSS_MOUNT_ROOT', '/data/oss_bucket_0')
        self.bucket = oss2.Bucket(
            oss2.Auth(os.environ['OSS_ACCESS_ID'], os.environ['OSS_ACCESS_KEY']),
            os.environ['OSS_ENDPOINT'], self.bucket_name, connect_timeout=30)

    def key(self, path):
        return object_key(path, self.bucket_name, self.mount_root)

    def head(self, key):
        try:
            return self.bucket.head_object(key).content_length
        except self.oss2.exceptions.NotFound:
            return None

    def inventory(self, root):
        prefix = root.rstrip('/') + '/'
        objects = {}
        marker = ''
        while True:
            page = self.bucket.list_objects(prefix=prefix, marker=marker, max_keys=1000)
            objects.update((obj.key, obj.size) for obj in page.object_list)
            if not page.is_truncated:
                return objects
            if not page.next_marker or page.next_marker == marker:
                raise PreflightError('OSS listing did not advance')
            marker = page.next_marker

    def read(self, key):
        size = self.head(key)
        if size is None or size <= 0 or size > 4 * 1024 * 1024:
            raise PreflightError('Missing, empty, or oversized metadata: ' + key)
        result = self.bucket.get_object(key)
        try:
            return result.read()
        finally:
            result.resp.response.close()

    def unused(self, key, directory=True):
        prefix = key.rstrip('/') + ('/' if directory else '.')
        if self.head(key) is not None or self.bucket.list_objects(prefix=prefix, max_keys=1).object_list:
            raise PreflightError('Output or submission receipt already exists: ' + key)

    def put_new(self, key, source):
        self.bucket.put_object(key, Path(source).read_bytes(),
                               headers={'x-oss-forbid-overwrite': 'true'})


def require_objects(objects, keys):
    for key in keys:
        if objects.get(key, 0) <= 0:
            raise PreflightError('Required OSS object missing or empty: ' + key)


def validate_manifest(data, root, objects):
    """Check manifest syntax and object presence; workers verify actual file hashes."""
    entries = {}
    try:
        lines = data.decode('utf-8').splitlines()
    except UnicodeDecodeError:
        raise PreflightError('Checksum manifest is not UTF-8') from None
    for line in lines:
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        if not match:
            raise PreflightError('Invalid checksum manifest line')
        digest, relative = match.groups()
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise PreflightError('Checksum path escapes its artifact directory')
        key = root.rstrip('/') + '/' + str(path)
        if key in entries:
            raise PreflightError('Duplicate checksum manifest entry')
        entries[key] = digest.lower()
    if not entries:
        raise PreflightError('Empty checksum manifest')
    require_objects(objects, entries)
    return entries


def preflight(client, args):
    checkpoint = client.key(args.checkpoint)
    assets = client.key(args.assets)
    for path in args.archive:
        key = client.key(path)
        if not client.head(key):
            raise PreflightError('Required archive missing or empty: ' + key)
    checkpoint_objects = client.inventory(checkpoint)
    assets_objects = client.inventory(assets)
    norm = checkpoint + '/assets/' + args.repo_id + '/norm_stats.json'
    model_files = (['model.safetensors', 'metadata.json'] if args.model_format == 'torch'
                   else ['params/_METADATA'])
    require_objects(checkpoint_objects, [norm] + [checkpoint + '/' + name for name in model_files])
    excluded = 'params/_METADATA' if args.model_format == 'torch' else 'model.safetensors'
    if checkpoint + '/' + excluded in checkpoint_objects:
        raise PreflightError('Checkpoint contains both JAX and Torch formats')
    require_objects(assets_objects, [assets + '/' + name for name in (
        'SHA256SUMS', 'behavior-1k-assets-3.9.0.zip', 'omnigibson-robot-assets-3.8.2.zip',
        '2026-challenge-task-instances.zip', 'omnigibson.key')])
    asset_entries = validate_manifest(client.read(assets + '/SHA256SUMS'), assets, assets_objects)
    if any(assets + '/' + name not in asset_entries for name in (
            'behavior-1k-assets-3.9.0.zip', 'omnigibson-robot-assets-3.8.2.zip',
            '2026-challenge-task-instances.zip', 'omnigibson.key')):
        raise PreflightError('Asset checksum manifest does not cover required asset files')
    ready = {}
    if args.staged:
        require_objects(checkpoint_objects, [checkpoint + '/' + name for name in (
            '_CHECKPOINT_METADATA', 'SHA256SUMS.inference', '_READY')])
        manifest = client.read(checkpoint + '/SHA256SUMS.inference')
        checkpoint_entries = validate_manifest(manifest, checkpoint, checkpoint_objects)
        required_covered = [norm] + [checkpoint + '/' + name for name in model_files]
        if any(key not in checkpoint_entries for key in required_covered):
            raise PreflightError('Inference checksum manifest does not cover required model files')
        try:
            ready = json.loads(client.read(checkpoint + '/_READY'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise PreflightError('Checkpoint _READY is not valid JSON') from None
        if not isinstance(ready, dict) or ready.get('manifest_sha256') != hashlib.sha256(manifest).hexdigest():
            raise PreflightError('Checkpoint _READY does not match its inference manifest')
    elif checkpoint_objects.get(checkpoint + '/_READY', 0):
        ready = json.loads(client.read(checkpoint + '/_READY'))
        if not isinstance(ready, dict):
            raise PreflightError('Checkpoint _READY must be a JSON object')
    client.unused(client.key(args.output))
    client.unused(client.key(args.receipt_prefix), directory=False)
    Path(args.ready_out).write_text(json.dumps(ready) + '\n')
    print('OSS preflight passed: archives, checkpoint, assets, and unused output/receipt paths.')
    print('Asset manifest objects:', len(asset_entries))
    print('Full SHA256 content verification runs in the Nebula worker before evaluation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('preflight')
    for name in ('checkpoint', 'assets', 'repo-id', 'output', 'receipt-prefix', 'ready-out'):
        check.add_argument('--' + name, required=True)
    check.add_argument('--archive', action='append', required=True)
    check.add_argument('--model-format', choices=('jax', 'torch'), required=True)
    check.add_argument('--staged', action='store_true')
    reserve = commands.add_parser('reserve')
    reserve.add_argument('--source', required=True)
    reserve.add_argument('--destination', required=True)
    sync = commands.add_parser('sync')
    sync.add_argument('--receipt', required=True)
    sync.add_argument('--stdout', required=True)
    sync.add_argument('--prefix', required=True)
    args = parser.parse_args()
    try:
        client = OSSClient()
        if args.command == 'preflight':
            preflight(client, args)
        elif args.command == 'reserve':
            client.put_new(client.key(args.destination), args.source)
            print('OSS submission reservation saved.')
        else:
            client.put_new(client.key(args.prefix) + '.json', args.receipt)
            client.put_new(client.key(args.prefix) + '.stdout.log', args.stdout)
            print('Submission receipt and CLI log synced to OSS.')
    except PreflightError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        # SDK exception strings can include request details. Do not print them.
        status = getattr(error, 'status', None)
        print('OSS operation failed (' + type(error).__name__ +
              (', HTTP ' + str(status) if isinstance(status, int) else '') + '). '
              'No submission retry was attempted.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

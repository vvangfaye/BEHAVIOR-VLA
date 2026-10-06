#!/usr/bin/env bash
# Copy to nebula/config.sh (git ignored), then chmod 600 nebula/config.sh.
# Credentials remain in the existing CLI stores; environment overrides take priority.
set +x

_nebula_exports="$("${NEBULA_CONFIG_PYTHON:-python3}" - <<'PY'
import configparser
import json
import os
from pathlib import Path
import re
import shlex
import sys


def read_aihub(path):
    """Read the flat string configuration written by ai-hub-cli config."""
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'(namespace|token):\s*(.*?)\s*', line)
        if not match:
            raise ValueError('Unsupported AI-Hub config format')
        key, value = match.groups()
        if value.startswith('"'):
            value = json.loads(value)
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1].replace("''", "'")
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0]
        if not isinstance(value, str) or key in result:
            raise ValueError('Unsupported AI-Hub config format')
        result[key] = value
    return result


try:
    aihub = {}
    if not all(os.environ.get(k) for k in ('AI_HUB_TOKEN', 'AI_HUB_NAMESPACE')):
        aihub = read_aihub(Path(os.environ.get(
            'AI_HUB_CONFIG_PATH', str(Path.home() / '.ai-hub-cli/config.yaml'))))
    oss = {}
    if not all(os.environ.get(k) for k in ('OSS_ACCESS_ID', 'OSS_ACCESS_KEY')):
        parser = configparser.ConfigParser(interpolation=None)
        with Path(os.environ.get(
                'OSSUTIL_CONFIG_PATH', str(Path.home() / '.ossutilconfig'))).open() as stream:
            parser.read_file(stream)
        oss = parser['Credentials']
    values = {
        'AI_HUB_TOKEN': os.environ.get('AI_HUB_TOKEN') or aihub.get('token', ''),
        'AI_HUB_NAMESPACE': os.environ.get('AI_HUB_NAMESPACE') or aihub.get('namespace', ''),
        'OSS_ACCESS_ID': os.environ.get('OSS_ACCESS_ID') or oss.get('accesskeyid', ''),
        'OSS_ACCESS_KEY': os.environ.get('OSS_ACCESS_KEY') or oss.get('accesskeysecret', ''),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError('Missing credentials')
    # Emit only after every field passes validation. The shell consumes this privately.
    for key, value in values.items():
        print('export ' + key + '=' + shlex.quote(value))
except Exception:
    print('Nebula credentials could not be loaded. Check AI-Hub/OSS CLI config files '
          'or set AI_HUB_TOKEN, AI_HUB_NAMESPACE, OSS_ACCESS_ID, OSS_ACCESS_KEY.', file=sys.stderr)
    sys.exit(1)
PY
)" || { unset _nebula_exports; return 1 2>/dev/null || exit 1; }
eval "$_nebula_exports"
unset _nebula_exports

# Pin this project location independently of unrelated global OSS settings.
export OSS_BUCKET="${NEBULA_OSS_BUCKET:-traffic-model}"
export OSS_ENDPOINT="${NEBULA_OSS_ENDPOINT:-http://cn-shanghai.oss.aliyun-inc.com}"
export QUEUE="${QUEUE:-aios_specific_nm125_l20}"
export GPU_CARD_TYPE="${GPU_CARD_TYPE:-l20}"
export ALGO_NAME="${ALGO_NAME:-pytorch2120}"
if ! command -v "${AI_HUB_CLI:-ai-hub-cli}" >/dev/null 2>&1; then
    if [[ -x "$HOME/.local/bin/ai-hub-cli" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo 'ai-hub-cli not found; install it or set AI_HUB_CLI.' >&2
        return 1 2>/dev/null || exit 1
    fi
fi
export AI_HUB_CLI="${AI_HUB_CLI:-ai-hub-cli}"

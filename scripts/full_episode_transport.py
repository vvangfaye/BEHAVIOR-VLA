"""Bounded object transport; never retry an episode or expose signed URLs."""
import json
import random
import ssl
import time
import urllib.error
import urllib.request


class ObjectTransportError(RuntimeError):
    pass


def safe_error(error):
    reason = getattr(error, 'reason', error)
    return dict(error_type=type(error).__name__, reason_type=type(reason).__name__,
                errno=getattr(reason, 'errno', None),
                http_status=getattr(error, 'code', None),
                tls_verify_code=getattr(reason, 'verify_code', None))


def request(url, *, method='GET', body=None, headers=None):
    # GET/HEAD and identical numbered-part PUT are repeatable. Completion POST
    # has an ambiguous commit boundary and must instead be reconciled by HEAD.
    attempts = 1 if method == 'POST' else 6
    deadline = time.monotonic() + 600
    last = None
    for attempt in range(attempts):
        try:
            if hasattr(body, 'seek'):
                body.seek(0)
            timeout = max(1, min(180, deadline - time.monotonic()))
            with urllib.request.urlopen(urllib.request.Request(
                    url, data=body, method=method, headers=headers or {}), timeout=timeout) as response:
                return response.read(), {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError('Collection object not found') from None
            last = safe_error(error)
            retryable = error.code in (408, 429, 500, 502, 503, 504)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            last = safe_error(error)
            reason = getattr(error, 'reason', error)
            retryable = not isinstance(reason, ssl.SSLCertVerificationError)
        if not retryable or attempt + 1 >= attempts or time.monotonic() >= deadline:
            break
        delay = min(2 ** (attempt + 1), 30) + random.SystemRandom().uniform(0, 1)
        if time.monotonic() + delay >= deadline:
            break
        print('OBJECT_TRANSPORT_RETRY=' + json.dumps(dict(
            method=method, attempt=attempt + 1, delay_seconds=round(delay, 2), **last)), flush=True)
        time.sleep(delay)
    raise ObjectTransportError('Collection object operation failed: ' + json.dumps(
        dict(method=method, **last), sort_keys=True)) from None

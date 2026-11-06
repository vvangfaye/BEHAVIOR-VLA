#!/usr/bin/env python3
"""Minimal AWS SigV4 client for AOSS using only the standard library (2026-11-06; cluster images have no boto3).
Path-style requests to https://<host>/<bucket>/<key>. Credentials come from the environment
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) and are never printed.
  get_sha256(key)          stream an object, return (sha256 hex, length)
  put_bytes(key, body)     PUT a small object
  get_bytes(key)           GET a small object
  delete(key)              DELETE an object
  list_keys(prefix)        ListObjectsV2 (all pages) -> [(key, size)]"""
import datetime
import hashlib
import hmac
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

REGION = os.environ.get("AOSS_REGION", "cn-sh-01b")
HOST = os.environ.get("AOSS_HOST", "aoss-internal.cn-sh-01b.sensecoreapi-oss.cn")
BUCKET = os.environ.get("AOSS_BUCKET", "260010165")
EMPTY = hashlib.sha256(b"").hexdigest()


def _sign(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _request(method, key, body=b"", query=None, timeout=900):
    ak, sk = os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"]
    now = datetime.datetime.utcnow()
    amz, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    uri = "/" + urllib.parse.quote(BUCKET, safe="") + ("/" + urllib.parse.quote(key, safe="/~") if key else "")
    q = sorted((query or {}).items())
    cq = "&".join("%s=%s" % (urllib.parse.quote(k, safe="-_.~"), urllib.parse.quote(str(v), safe="-_.~")) for k, v in q)
    ph = hashlib.sha256(body).hexdigest() if body else EMPTY
    headers = {"host": HOST, "x-amz-content-sha256": ph, "x-amz-date": amz}
    sh = ";".join(sorted(headers))
    creq = "\n".join([method, uri, cq, "".join("%s:%s\n" % (h, headers[h]) for h in sorted(headers)), sh, ph])
    scope = "%s/%s/s3/aws4_request" % (day, REGION)
    sts = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(creq.encode()).hexdigest()])
    k = _sign(_sign(_sign(_sign(("AWS4" + sk).encode(), day), REGION), "s3"), "aws4_request")
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request("https://%s%s%s" % (HOST, uri, ("?" + cq) if cq else ""), data=body if body else None, method=method)
    for h, v in headers.items():
        if h != "host":
            req.add_header(h, v)
    req.add_header("Authorization", "AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s" % (ak, scope, sh, sig))
    return urllib.request.urlopen(req, timeout=timeout)


def get_sha256(key):
    h, n = hashlib.sha256(), 0
    with _request("GET", key) as r:
        while True:
            b = r.read(8 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def get_bytes(key):
    with _request("GET", key) as r:
        return r.read()


def put_bytes(key, body):
    with _request("PUT", key, body=body) as r:
        return r.status


def delete(key):
    with _request("DELETE", key) as r:
        return r.status


def list_keys(prefix):
    out, token = [], None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with _request("GET", "", query=q) as r:
            root = ET.fromstring(r.read())
        for c in root.iter(ns + "Contents"):
            out.append((c.find(ns + "Key").text, int(c.find(ns + "Size").text)))
        if (root.findtext(ns + "IsTruncated") or "false").lower() != "true":
            return out
        token = root.findtext(ns + "NextContinuationToken")

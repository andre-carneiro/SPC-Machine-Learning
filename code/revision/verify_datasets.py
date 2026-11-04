"""Check the downloaded datasets (data/processed/*.parquet) against the SHA-256 checksums of manifest.json."""
import hashlib
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


manifest = json.load(open(os.path.join(ROOT, "data", "processed", "manifest.json")))
status = {}
for name, d in manifest["datasets"].items():
    path = os.path.join(ROOT, d["file"])
    status[name] = "missing" if not os.path.exists(path) else ("ok" if sha256(path) == d["sha256"] else "CHECKSUM MISMATCH")
    print(f"{name:20s} {status[name]}")
print(f"{sum(v == 'ok' for v in status.values())}/{len(status)} datasets verified")

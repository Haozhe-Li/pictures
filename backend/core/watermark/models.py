"""
TrustMark's ONNX models: where they live, and fetching them once, checked against Adobe's MD5s.

The Docker image fetches them while it is built (`python -m core.watermark`), so the server never
waits on Adobe's CDN; anywhere else (a dev machine) they are fetched on first use.
"""
import hashlib
import os
import tempfile
import urllib.request

from core.config import settings

# Adobe's CDN for the models (python-hailo/trustmark_hailo/trustmark_hailo.py, commit 6df5a0a)
MODEL_REMOTE_HOST = "https://cai-watermark.adobe.net/watermarking/trustmark-models/"

MODEL_CHECKSUMS = {
    "encoder_Q.onnx": "d6fa92d4b4fe8d67ad196b2d21320955",
    "decoder_Q.onnx": "90427159004ddbf5128271ba0684cf42",
}


def model_name(part: str) -> str:
    """"encoder" or "decoder" -> the file for the gallery's variant, e.g. "encoder_Q.onnx"."""
    return f"{part}_{settings.WATERMARK_MODEL_TYPE}.onnx"


def _md5(path: str) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Files already checked by this process (each image loads its model again)
_checked = set()


def ensure_model(name: str) -> str:
    """The path of model file `name`, fetched first if it is missing or not the expected file."""
    expected = MODEL_CHECKSUMS[name]
    path = os.path.join(settings.WATERMARK_MODEL_DIR, name)
    if path in _checked or (os.path.isfile(path) and _md5(path) == expected):
        _checked.add(path)
        return path

    print(f"Fetching TrustMark model {name}...")
    os.makedirs(settings.WATERMARK_MODEL_DIR, exist_ok=True)
    # Written beside the target and moved into place, so a cut-off download never looks finished
    fd, partial = tempfile.mkstemp(dir=settings.WATERMARK_MODEL_DIR, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(MODEL_REMOTE_HOST + name, timeout=60) as res:
            for chunk in iter(lambda: res.read(1 << 20), b""):
                out.write(chunk)
        actual = _md5(partial)
        if actual != expected:
            raise RuntimeError(f"{name} has MD5 {actual}, expected {expected}")
        os.chmod(partial, 0o644)  # mkstemp makes it private to whoever fetched it
        os.replace(partial, path)
        _checked.add(path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return path

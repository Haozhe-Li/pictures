"""
Invisible watermarks with Adobe's TrustMark (https://github.com/adobe/trustmark), variant Q, run on
ONNX Runtime with numpy and Pillow (no PyTorch).

Nothing stays in memory: each image loads the one model it needs (the encoder to watermark a photo, the
decoder to read one; ~0.05 s) and drops it with everything else it used as soon as it is done, handing
the memory back to the system. One image is processed at a time. The server is small: models kept
around, or added on top of a full-size photo's arithmetic, are what pushed an upload out of memory.

A watermarked photo carries a random 61-bit id (TrustMark's payload with its default BCH_5 error
correction), stored on its record as `watermark_id`, so a mark found later leads back to the photo.
"""
import ctypes
import gc
import io
import secrets
import threading
from contextlib import contextmanager
from typing import Optional

from PIL import Image

from core.config import settings
from .models import ensure_model, model_name

ID_BITS = 61
# The DataLayer schema of the marks written here (BCH_5). Random bits now and then pass as one of the
# weaker schemas (BCH_3 corrects 3 flips in 96 bits), so only this one counts as a mark.
SCHEMA = 1

_TURNS = (None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)


class WatermarkUnavailable(RuntimeError):
    """The models could not be fetched or loaded."""


def new_watermark_id() -> int:
    while True:
        value = secrets.randbits(ID_BITS)
        # All zeros is a valid codeword, and what a blank decoder output would read as
        if value:
            return value


def format_watermark_id(value: int) -> str:
    """The form `watermark_id` is stored and shown in: 16 hex digits."""
    return f"{value:016x}"


class Watermarker:
    def __init__(self, threads: int):
        self.threads = threads
        self._sessions = {}
        self._core = None
        # One image at a time; also guards loading and dropping the models
        self._lock = threading.Lock()

    @property
    def loaded(self) -> list:
        """The models in memory right now ("decoder", "encoder"); only ever while an image is processed."""
        return sorted(self._sessions)

    def embed(self, image: Image.Image, watermark_id: int) -> Image.Image:
        """`image` (RGB) with `watermark_id` in it, the same size."""
        with self._job():
            secret = format(watermark_id, f"0{ID_BITS}b")
            return self._trustmark().encode(image, secret, MODE="binary")

    def read(self, image: Image.Image, rotations: bool = False) -> Optional[int]:
        """The id in `image` (RGB), or None. With `rotations`, it is also tried turned by 90, 180 and 270 degrees."""
        with self._job():
            core = self._trustmark()
            for turn in _TURNS if rotations else _TURNS[:1]:
                candidate = image if turn is None else image.transpose(turn)
                bits, detected, schema = core.subimage_decode(core.get_the_image_for_processing(candidate))
                if detected and schema == SCHEMA and int(bits, 2):
                    return int(bits, 2)
        return None

    def release(self) -> None:
        """Drops the models now."""
        with self._lock:
            self._drop()

    # --- Loading and dropping --------------------------------------------------------------------

    @contextmanager
    def _job(self):
        with self._lock:
            try:
                yield
            finally:
                self._drop()

    def _trustmark(self):
        if self._core is None:
            from .trustmark import TrustMarkCore

            self._core = TrustMarkCore(
                self._encoder_infer, self._decoder_infer, model_type=settings.WATERMARK_MODEL_TYPE
            )
        return self._core

    def _session(self, part: str):
        session = self._sessions.get(part)
        if session is None:
            try:
                import onnxruntime as ort

                path = ensure_model(model_name(part))
                options = ort.SessionOptions()
                options.intra_op_num_threads = self.threads
                options.inter_op_num_threads = 1
                # Give memory back after each run rather than keeping a pool for the next one
                options.enable_cpu_mem_arena = False
                # The layout optimizations of ORT_ENABLE_ALL keep a second, repacked copy of the weights:
                # the decoder takes ~210 MB with them and ~65 MB without, for ~80 ms instead of ~30 ms a run
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
            except Exception as e:
                raise WatermarkUnavailable(f"could not load the TrustMark {part}: {e}") from e
            self._sessions[part] = session
        return session

    def _encoder_infer(self, cover_nchw, secret):
        session = self._session("encoder")
        cover, payload = [i.name for i in session.get_inputs()]
        stego = session.run(None, {cover: cover_nchw, payload: secret})[0]
        # It runs once an image, and what follows is arithmetic on the full-size photo: hand the model and
        # its working memory back first, so the two never add up
        del session
        self._drop()
        return stego

    def _decoder_infer(self, image_nchw):
        session = self._session("decoder")
        return session.run(None, {session.get_inputs()[0].name: image_nchw})[0]

    def _drop(self) -> None:
        self._sessions.clear()
        gc.collect()
        _trim_heap()


def _trim_heap() -> None:
    """Hands freed memory back to the system; glibc otherwise keeps ~100 MB of what an image used."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc


watermarker = Watermarker(settings.WATERMARK_THREADS)

# The decoder looks at 256 x 256 pixels; reading a smaller copy spares decoding a large photo at full size
READ_MAX_SIZE = 2048


def watermark_photo(file_bytes: bytes, watermark_id: int, max_size: int) -> Image.Image:
    """
    An uploaded photo at its published size (long edge `max_size`) with `watermark_id` in it.

    Scaling first keeps the float math to the size that is published (a full 24 MP frame would take
    about 1 GB). Transparency, if any, is kept as it was.
    """
    image = Image.open(io.BytesIO(file_bytes))
    image.thumbnail((max_size, max_size))
    alpha = image.getchannel("A") if image.mode == "RGBA" else None
    marked = watermarker.embed(image if image.mode == "RGB" else image.convert("RGB"), watermark_id)
    if alpha is not None:
        marked.putalpha(alpha)
    return marked


def read_photo(file_bytes: bytes, rotations: bool = False) -> Optional[int]:
    """The watermark id in an image file, or None. Raises PIL's errors for a file that is not an image."""
    with Image.open(io.BytesIO(file_bytes)) as image:
        image.thumbnail((READ_MAX_SIZE, READ_MAX_SIZE))
        return watermarker.read(image if image.mode == "RGB" else image.convert("RGB"), rotations=rotations)

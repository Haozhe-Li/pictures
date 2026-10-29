# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

# Vendored from adobe/trustmark, python-hailo/trustmark_hailo/pipeline.py (commit 6df5a0a, MIT, see
# LICENSE): TrustMark's encode/decode around two injected forward passes, with numpy and Pillow only.
# Trimmed to what the gallery uses: the bounding-box detector (localize, DETECTFIRST) and decode() are left
# out; core/watermark/__init__.py reads marks with subimage_decode().

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from .datalayer import DataLayer

CONCENTRATE_WM_REGION = 1.0
ASPECT_RATIO_LIM = 2.0
FEATHERING_RESIDUAL = 0.01

# Both ONNX graphs take/return NCHW float32 in [-1, 1] at 256x256.
MODEL_RESOLUTION = 256


def pil_to_model_input(img: Image.Image, resolution: int = MODEL_RESOLUTION) -> np.ndarray:
    """PIL RGB image -> (1,3,res,res) float32 NCHW in [-1, 1]."""
    resized = img.resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0 * 2.0 - 1.0
    return arr.transpose(2, 0, 1)[None].copy()


def model_output_to_stego_arr(stego_nchw: np.ndarray) -> np.ndarray:
    """(1,3,H,W) float32 in [-1,1] -> (H,W,3) float32 in [-1,1]."""
    return stego_nchw[0].transpose(1, 2, 0)


# EncoderInfer: (cover_nchw[-1,1] float32 (1,3,256,256), secret (1,secret_len) float32 0/1) -> stego_nchw (1,3,256,256) float32
EncoderInfer = Callable[[np.ndarray, np.ndarray], np.ndarray]
# DecoderInfer: (image_nchw[-1,1] float32 (1,3,H,W)) -> logits (1,secret_len) float32.
# H,W is 256 for variants C/Q/B (their decoder graphs resize internally), 224 for variant P.
DecoderInfer = Callable[[np.ndarray], np.ndarray]


def _resize_float_hwc(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an (H,W,3) float array (arbitrary range) with bilinear
    interpolation, channel-by-channel via PIL's 'F' mode (avoids requiring
    torch or opencv just for this one resize)."""
    w, h = size
    out = np.empty((h, w, arr.shape[2]), dtype=np.float32)
    for c in range(arr.shape[2]):
        # A 2-D float32 array is mode "F"; Pillow 11.3 deprecated passing the mode
        band = Image.fromarray(arr[:, :, c].astype(np.float32))
        out[:, :, c] = np.asarray(band.resize((w, h), Image.BILINEAR))
    return out


class TrustMarkCore:
    """Same encode()/decode() contract as trustmark.TrustMark, but the two
    neural net forward passes are delegated to injected callables so this
    class has no PyTorch dependency of its own."""

    def __init__(
        self,
        encoder_infer: EncoderInfer,
        decoder_infer: DecoderInfer,
        secret_len: int = 100,
        encoding_type: int = 1,  # DataLayer/BCH schema id; 1 == BCH_5 (61-bit payload), matches TrustMark's default
        model_type: str = "Q",
        concentrate_wm_region: float = CONCENTRATE_WM_REGION,
    ):
        self.encoder_infer = encoder_infer
        self.decoder_infer = decoder_infer
        self.secret_len = secret_len
        self.ecc = DataLayer(secret_len, encoding_mode=encoding_type)
        self.model_type = model_type
        self.concentrate_wm_region = concentrate_wm_region
        self.encoder_resolution = MODEL_RESOLUTION  # 256 for every variant (C/Q/B/P)
        self.aspect_ratio_lim = ASPECT_RATIO_LIM

        if model_type == "P":
            self.decoder_resolution = 224
            self.aspect_ratio_lim = 0  # always force to centre square crop
        else:
            self.decoder_resolution = MODEL_RESOLUTION

    def schemaCapacity(self) -> int:
        return self.ecc.schemaCapacity(self.ecc.encoding_mode)

    def get_the_image_for_processing(self, in_image: Image.Image) -> Image.Image:
        scale = self.concentrate_wm_region
        width, height = in_image.size

        # Compute aspect ratio (>= 1.0)
        if width > height:
            aspect_ratio = width / height
        else:
            aspect_ratio = height / width

        # Make a copy of the image (PIL)
        out_im = in_image.copy()

        if aspect_ratio > self.aspect_ratio_lim:
            # We do a center-square approach, but scaled
            square_size = min(width, height)  # largest possible square dimension
            scaled_size = int(square_size * scale)  # scale that dimension

            # Compute bounding box
            left = (width - scaled_size) // 2
            top = (height - scaled_size) // 2
            right = left + scaled_size
            bottom = top + scaled_size

            out_im = out_im.crop((left, top, right, bottom))

        else:
            # The aspect ratio is normal, so we consider
            # the *entire* image dimension. Then scale that region
            scaled_w = int(width * scale)
            scaled_h = int(height * scale)

            # Center the smaller (or bigger) rectangle
            left = (width - scaled_w) // 2
            top = (height - scaled_h) // 2
            right = left + scaled_w
            bottom = top + scaled_h

            out_im = out_im.crop((left, top, right, bottom))

        return out_im

    def put_the_image_after_processing(self, wm_image: np.ndarray, cover_im: np.ndarray, feather: bool = True) -> np.ndarray:
        scale = self.concentrate_wm_region
        cover_h, cover_w, _ = cover_im.shape

        if cover_w > cover_h:
            aspect_ratio = cover_w / cover_h
        else:
            aspect_ratio = cover_h / cover_w

        out_im = cover_im.copy()

        if aspect_ratio > self.aspect_ratio_lim:
            # Square region, scaled
            square_size = min(cover_w, cover_h)
            scaled_size = int(square_size * scale)

            left = (cover_w - scaled_size) // 2
            top = (cover_h - scaled_size) // 2
            right = left + scaled_size
            bottom = top + scaled_size

            region_w = scaled_size
            region_h = scaled_size

        else:
            # Normal ratio, scaled
            scaled_w = int(cover_w * scale)
            scaled_h = int(cover_h * scale)
            left = (cover_w - scaled_w) // 2
            top = (cover_h - scaled_h) // 2
            right = left + scaled_w
            bottom = top + scaled_h

            region_w = scaled_w
            region_h = scaled_h

        if feather:
            feather_size = int(min(region_w, region_h) * FEATHERING_RESIDUAL)

            feather_size = max(1, feather_size)
            feather_size = min(feather_size, 50)

            self.feather_paste(out_im, cover_im, wm_image, top, bottom, left, right, feather_size=feather_size)
        else:
            out_im[top:bottom, left:right, :] = wm_image

        return out_im

    def feather_paste(
        self,
        out_im: np.ndarray,  # Output image (modified in-place)
        cover_im: np.ndarray,  # Original cover image (same shape)
        wm_image: np.ndarray,  # Watermarked patch to paste
        top: int,
        bottom: int,
        left: int,
        right: int,
        feather_size: int = 9,
    ) -> None:
        out_im[top:bottom, left:right, :] = wm_image
        alpha_vals = [(i + 1) / feather_size for i in range(feather_size)]

        feather_size = min(feather_size, (bottom - top), (right - left))

        for i in range(feather_size):
            alpha = alpha_vals[i]
            row = top + i
            # Blend that entire row from left..right
            out_im[row, left:right, :] = alpha * wm_image[i, :, :] + (1.0 - alpha) * cover_im[row, left:right, :]

        for i in range(feather_size):
            alpha = alpha_vals[i]
            row = bottom - 1 - i
            wm_row = (bottom - top - 1) - i
            out_im[row, left:right, :] = alpha * wm_image[wm_row, :, :] + (1.0 - alpha) * cover_im[row, left:right, :]

        for i in range(feather_size):
            alpha = alpha_vals[i]
            col = left + i
            out_im[top:bottom, col, :] = alpha * wm_image[:, i, :] + (1.0 - alpha) * cover_im[top:bottom, col, :]

        for i in range(feather_size):
            alpha = alpha_vals[i]
            col = right - 1 - i
            wm_col = (right - left - 1) - i
            out_im[top:bottom, col, :] = alpha * wm_image[:, wm_col, :] + (1.0 - alpha) * cover_im[top:bottom, col, :]

    def encode(self, in_cover_image: Image.Image, string_secret: str, MODE: str = "binary", WM_STRENGTH: float = 1.0) -> Image.Image:
        if self.model_type == "P":
            WM_STRENGTH = WM_STRENGTH * 1.25

        secret = self.ecc.encode_binary([string_secret]) if MODE == "binary" else self.ecc.encode_text([string_secret])
        secret = secret.astype(np.float32)  # (1, secret_len), values in {0.0, 1.0}

        cover_image = self.get_the_image_for_processing(in_cover_image)
        w, h = cover_image.size
        cover_nchw = pil_to_model_input(cover_image, self.encoder_resolution)

        stego_nchw = self.encoder_infer(cover_nchw, secret)

        stego_hwc = model_output_to_stego_arr(np.clip(stego_nchw, -1, 1))
        cover_hwc = model_output_to_stego_arr(cover_nchw)
        residual = stego_hwc - cover_hwc
        residual = residual - residual.mean(axis=(0, 1), keepdims=True)  # remove per-channel color shift

        # Upstream: np.clip(residual_full * WM_STRENGTH + cover_arr / 127.5 - 1.0, -1, 1) * 127.5 + 127.5,
        # worked in place here so a full-size photo needs two float arrays at a time rather than five
        stego_full = _resize_float_hwc(residual, (w, h))
        stego_full *= WM_STRENGTH
        cover_arr = np.asarray(cover_image, dtype=np.float32)
        cover_arr /= 127.5
        stego_full += cover_arr
        del cover_arr
        stego_full -= 1.0
        np.clip(stego_full, -1, 1, out=stego_full)
        stego_full *= 127.5
        stego_full += 127.5

        stego = self.put_the_image_after_processing(stego_full, np.asarray(in_cover_image).astype(np.uint8))
        return Image.fromarray(stego.astype(np.uint8))

    def subimage_decode(self, stego_image: Image.Image, MODE: str = "binary"):
        stego_nchw = pil_to_model_input(stego_image, self.decoder_resolution)
        logits = self.decoder_infer(stego_nchw)
        secret_binaryarray = (logits > 0).astype(np.uint8)
        secret_pred, detected, version = self.ecc.decode_bitstream(secret_binaryarray, MODE)[0]
        return secret_pred, detected, version

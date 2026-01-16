import base64
import io
from PIL import Image


def process_image_for_embedding(file_bytes: bytes, max_size: int = 1024) -> str:
    """
    Resizes and compresses the image for embedding API consumption.
    Returns: Base64 string of the processed image (JPEG).
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(file_bytes))

        # Resize if dimensions exceed max_size (maintaining aspect ratio)
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))

        # Convert to RGB to ensure compatibility (e.g. removing Alpha channel for JPEG)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save to buffer as JPEG
        output = io.BytesIO()
        image.save(
            output, format="JPEG", quality=50
        )  # Quality 85 is usually good enough for embeddings

        # Encode to Base64
        return base64.b64encode(output.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error processing image: {e}")
        # In case of error, try to return original base64 or re-raise
        # For safety/fallback, let's just assume we re-raise because if PIL fails, the file is likely bad.
        raise ValueError(f"Failed to process image for embedding: {e}")


def save_as_webp(
    file_bytes: bytes,
    output_path: str,
    quality: int = 50,
    max_size: int | None = None,
) -> None:
    """
    Compresses image to WebP and saves to the specified path.
    Optionally downscales to max_size (longer edge).
    """
    image = Image.open(io.BytesIO(file_bytes))

    # Resize if dimensions exceed max_size (maintaining aspect ratio)
    if max_size and max(image.size) > max_size:
        image.thumbnail((max_size, max_size))

    # Ensure compatible mode for WebP (RGB or RGBA)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    image.save(output_path, format="WEBP", quality=quality)

import boto3
from botocore.config import Config
import os
import re
from typing import Optional
from core.config import settings

s3 = boto3.client(
    "s3",
    endpoint_url=settings.CF_API_URL,
    aws_access_key_id=settings.CF_API_KEY_ID,
    aws_secret_access_key=settings.CF_API_KEY_SECRET,
    config=Config(signature_version="s3v4"),
)


def upload_file_to_r2(file_path: str, file_name: str) -> str:
    """
    Uploads a file to Cloudflare R2 and returns the public URL.
    """

    bucket_name = settings.CF_BUCKET
    base_url = settings.CLOUDFLARE_FREE_URL

    try:
        s3.upload_file(file_path, bucket_name, file_name)
        url = f"{base_url}{file_name}"
        return url
    except Exception as e:
        print(f"Error uploading to R2: {e}")
        raise e


# --- Staging area (browser -> R2 direct uploads) ---
#
# The browser PUTs the raw file to a presigned URL under STAGING_PREFIX; the
# backend then pulls it down, processes it, publishes the final WebP files and
# deletes the staged object. Add an R2 lifecycle rule on this prefix (e.g.
# expire after 1 day) to sweep up uploads that were never ingested.

STAGING_PREFIX = "staging/"


def presign_staging_put(key: str, content_type: str, expires_in: int = 900) -> str:
    """
    Returns a presigned PUT URL for `key`. The signature covers Content-Type, so
    the browser must send exactly the same header.
    """
    return s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.CF_BUCKET,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=expires_in,
    )


def get_object_size(key: str) -> int:
    """Size in bytes of an object in the bucket. Raises ClientError if missing."""
    head = s3.head_object(Bucket=settings.CF_BUCKET, Key=key)
    return head["ContentLength"]


def download_object(key: str) -> bytes:
    """Reads an object from the bucket fully into memory."""
    obj = s3.get_object(Bucket=settings.CF_BUCKET, Key=key)
    return obj["Body"].read()


def delete_object(key: str) -> None:
    s3.delete_object(Bucket=settings.CF_BUCKET, Key=key)


# --- Published objects (what the gallery serves) ---

# Ingest always publishes `<uuid>_preview.webp` and `<uuid>_original.webp`. Deleting is
# restricted to that shape so a bad payload can never remove anything else in the bucket.
PUBLISHED_KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_(preview|original)\.webp$")


def key_from_public_url(url: Optional[str]) -> Optional[str]:
    """
    Maps a public CDN URL back to its bucket key, or None when the URL is not one of
    our published image files.
    """
    base_url = settings.CLOUDFLARE_FREE_URL
    if not url or not url.startswith(base_url):
        return None
    key = url[len(base_url):]
    return key if PUBLISHED_KEY_RE.match(key) else None

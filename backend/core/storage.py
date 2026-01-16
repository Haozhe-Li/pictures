import boto3
from botocore.config import Config
import os
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

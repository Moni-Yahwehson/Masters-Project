"""Image intake: validation and normalisation for both upload paths."""

from __future__ import annotations

import uuid
from typing import Iterable, Optional, Sequence

from fastapi import HTTPException, UploadFile, status

from .config import get_settings
from .llm import ImagePayload
from .schemas import InlineImage

ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"}

# Claude accepts jpeg/png/gif/webp. Anything else must be rejected up front
# rather than failing deep inside an agent.
_EXTENSION_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
}


def _new_id(index: int) -> str:
    return f"img-{index:02d}-{uuid.uuid4().hex[:8]}"


def _validate(payload: ImagePayload) -> ImagePayload:
    settings = get_settings()
    media_type = payload.media_type.lower()
    if media_type == "image/jpg":
        media_type = "image/jpeg"
        payload.media_type = media_type
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image type '{payload.media_type}'. "
                f"Use one of: {', '.join(sorted(ALLOWED_MEDIA_TYPES))}."
            ),
        )
    if payload.size_bytes > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image '{payload.filename or payload.id}' is "
                f"{payload.size_bytes // 1024} KB; the limit is "
                f"{settings.max_image_bytes // 1024} KB."
            ),
        )
    if payload.size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image '{payload.filename or payload.id}' is empty.",
        )
    return payload


def _enforce_count(count: int) -> None:
    settings = get_settings()
    if count > settings.max_images_per_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Too many images: {count}. "
                f"The limit is {settings.max_images_per_request} per request."
            ),
        )


def decode_inline_images(images: Sequence[InlineImage]) -> list[ImagePayload]:
    """Decode base64 data URLs, which is what the Vue client sends."""
    _enforce_count(len(images))
    payloads: list[ImagePayload] = []
    for index, image in enumerate(images, start=1):
        try:
            payload = ImagePayload.from_data_url(
                image.data_url, image_id=_new_id(index), filename=image.filename
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Attachment {index} is not a valid base64 data URL: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - malformed base64
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Attachment {index} could not be decoded: {exc}",
            ) from exc
        payloads.append(_validate(payload))
    return payloads


async def decode_uploads(files: Iterable[UploadFile]) -> list[ImagePayload]:
    """Decode a multipart/form-data upload."""
    file_list = [file for file in files if file is not None]
    _enforce_count(len(file_list))
    payloads: list[ImagePayload] = []
    for index, upload in enumerate(file_list, start=1):
        raw = await upload.read()
        media_type = _resolve_media_type(upload)
        payloads.append(
            _validate(
                ImagePayload.from_bytes(
                    raw,
                    image_id=_new_id(index),
                    media_type=media_type,
                    filename=upload.filename,
                )
            )
        )
    return payloads


def _resolve_media_type(upload: UploadFile) -> str:
    declared = (upload.content_type or "").lower()
    if declared in ALLOWED_MEDIA_TYPES:
        return "image/jpeg" if declared == "image/jpg" else declared
    extension = (upload.filename or "").rsplit(".", 1)
    if len(extension) == 2:
        guessed: Optional[str] = _EXTENSION_MEDIA_TYPES.get(extension[1].lower())
        if guessed:
            return guessed
    return declared or "application/octet-stream"

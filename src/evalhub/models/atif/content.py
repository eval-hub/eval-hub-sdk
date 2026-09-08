# Copyright 2025 The Harbor Project Authors
# Copyright 2025 Alex Shaw
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: This file contains code adapted from Harbor
# (https://github.com/harbor-framework/harbor) and has been modified
# to isolate Pydantic models as a standalone dependency.

"""Content models for multimodal ATIF trajectories.

Added in ATIF-v1.6 to support multimodal content (images) in trajectories.
Extended in ATIF-v1.8 with audio.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Canonical audio MIME types, chosen to cover what multimodal and transcription
# APIs actually accept: Gemini audio understanding (wav, mp3, aiff, aac, ogg,
# flac), OpenAI transcription (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm),
# and `audio/webm`, which is what a browser's MediaRecorder produces and so what
# human-recorded task data usually arrives as.
AudioMediaType = Literal[
    "audio/wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "audio/aiff",
]

# Aliases producers emit in place of the registered MIME type, normalized on the
# way in so consumers only ever see one spelling per format. `audio/mp3` is the
# important one: it is not a registered type, but the Gemini API documents and
# emits it, so a producer copying a provider response verbatim would otherwise
# fail validation.
_AUDIO_MEDIA_TYPE_ALIASES = {
    "audio/mp3": "audio/mpeg",
    "audio/mpga": "audio/mpeg",
    "audio/x-mpeg": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/vnd.wave": "audio/wav",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
    "audio/x-aac": "audio/aac",
    "audio/x-flac": "audio/flac",
    "audio/x-aiff": "audio/aiff",
}

# Public input type: canonical values plus all accepted aliases. Widening the
# declared field type to include aliases lets callers pass them without a type
# error; the normalize_media_type validator maps everything to AudioMediaType
# before Pydantic's own Literal check runs.
AudioMediaTypeInput = (
    AudioMediaType
    | Literal[
        "audio/mp3",
        "audio/mpga",
        "audio/x-mpeg",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        "audio/x-m4a",
        "audio/m4a",
        "audio/x-aac",
        "audio/x-flac",
        "audio/x-aiff",
    ]
)


class ImageSource(BaseModel):
    """Image source specification for images stored as files or at remote URLs."""

    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = Field(
        default=...,
        description="MIME type of the image",
    )
    path: str = Field(
        default=...,
        description=(
            "Location of the image. Can be a relative or absolute file path, or a URL."
        ),
    )

    model_config = {"extra": "forbid"}


class AudioSource(BaseModel):
    """Audio source specification for audio stored as files or at remote URLs.

    Added in ATIF-v1.8. Mirrors ``ImageSource`` so that consumers can treat any
    referenced media the same way, with one addition: ``duration_sec``. Duration
    is worth recording because several providers bill audio per second and
    dataset statistics need it, and unlike an image's dimensions it cannot be
    recovered without decoding the file.
    """

    media_type: AudioMediaTypeInput = Field(
        default=...,
        description="MIME type of the audio",
    )
    path: str = Field(
        default=...,
        description=(
            "Location of the audio. Can be a relative or absolute file path, or a URL."
        ),
    )
    duration_sec: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional duration of the audio in seconds. Useful for cost "
            "accounting and dataset statistics; omit when unknown."
        ),
    )

    model_config = {"extra": "forbid"}

    @field_validator("media_type", mode="before")
    @classmethod
    def normalize_media_type(cls, value: Any) -> Any:
        """Accept common non-registered spellings, storing the canonical one."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            return _AUDIO_MEDIA_TYPE_ALIASES.get(normalized, normalized)
        return value


class ContentPart(BaseModel):
    """A single content part within a multimodal message.

    Used when a message or observation contains mixed content types (text,
    images, or audio). For text-only content, a plain string can still be used
    instead of a ContentPart array.
    """

    type: Literal["text", "image", "audio"] = Field(
        default=...,
        description="The type of content",
    )
    text: str | None = Field(
        default=None,
        description="Text content. Required when type='text'.",
    )
    source: ImageSource | AudioSource | None = Field(
        default=None,
        description=(
            "Media source (file reference). Required when type='image' or "
            "type='audio', and must match that type."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_content_type(self) -> "ContentPart":
        """Validate that the correct fields are present for each content type."""
        if self.type == "text":
            if self.text is None:
                raise ValueError("'text' field is required when type='text'")
            if self.source is not None:
                raise ValueError("'source' field is not allowed when type='text'")
            return self

        # Media parts: a source is required, text is not allowed, and the source
        # must be of the kind the `type` declares. Without the last check a
        # `source` union would happily accept an ImageSource under
        # type='audio' (and vice versa), silently mislabelling the content.
        if self.source is None:
            raise ValueError(f"'source' field is required when type='{self.type}'")
        if self.text is not None:
            raise ValueError(f"'text' field is not allowed when type='{self.type}'")

        expected = ImageSource if self.type == "image" else AudioSource
        if not isinstance(self.source, expected):
            raise ValueError(
                f"type='{self.type}' requires a {expected.__name__}, but the "
                f"source's media_type is {self.source.media_type!r}"
            )
        return self

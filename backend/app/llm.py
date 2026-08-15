"""Local LLM client — no external/hosted model APIs.

Every model call in this project runs on the machine that hosts the backend.
Inference is served by **Ollama**, which listens on `127.0.0.1:11434` and loads
open-weight models from local disk. Nothing leaves the host: there is no cloud
provider, no API key, and no per-token billing.

Why Ollama rather than in-process `transformers`/`llama-cpp-python`:

* it keeps the (multi-gigabyte) model out of the web worker's memory, so the
  API restarts in milliseconds and multiple workers share one loaded model;
* it exposes **JSON-schema-constrained decoding** (`format=<schema>`), which is
  what makes the structured agents reliable on small local models — the
  grammar makes malformed output impossible rather than merely unlikely;
* swapping the runtime means changing `LSD_LLM_BASE_URL`, not the code — any
  server speaking the same `/api/chat` shape works (e.g. `llama-server`
  behind a small shim).

Two models are used, both configurable:

* a **vision** model for the triage and lesion-analysis agents;
* a **text** model for synthesis, verification and knowledge Q&A.

They may be the same model if it is multimodal.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from .config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DATA_URL_RE = re.compile(r"^data:(?P<media_type>[\w/+.-]+);base64,(?P<data>.+)$", re.S)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.S)

#: How long a failed availability probe is cached before we retry the server.
#: Short enough that starting Ollama mid-session recovers without a restart.
_PROBE_TTL_SECONDS = 15.0


class LLMUnavailable(RuntimeError):
    """Raised when the local model could not produce a usable answer."""


class LLMNotRunning(LLMUnavailable):
    """Raised when the local inference server is not reachable."""


@dataclass
class ImagePayload:
    """A decoded image ready to be handed to the vision model."""

    id: str
    media_type: str
    base64_data: str
    filename: Optional[str] = None
    size_bytes: int = 0

    @classmethod
    def from_data_url(
        cls, data_url: str, image_id: str, filename: Optional[str] = None
    ) -> "ImagePayload":
        match = _DATA_URL_RE.match(data_url.strip())
        if not match:
            raise ValueError("Attachment is not a base64 data URL")
        media_type = match.group("media_type")
        payload = match.group("data").replace("\n", "").replace("\r", "")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Attachment is not valid base64") from exc
        return cls(
            id=image_id,
            media_type=media_type,
            base64_data=base64.b64encode(raw).decode("ascii"),
            filename=filename,
            size_bytes=len(raw),
        )

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        image_id: str,
        media_type: str = "image/jpeg",
        filename: Optional[str] = None,
    ) -> "ImagePayload":
        return cls(
            id=image_id,
            media_type=media_type,
            base64_data=base64.b64encode(raw).decode("ascii"),
            filename=filename,
            size_bytes=len(raw),
        )

    def to_bytes(self) -> bytes:
        return base64.b64decode(self.base64_data)

    def to_content_block(self) -> dict[str, Any]:
        return {"type": "image", "image_id": self.id, "data": self.base64_data}


def image_blocks(images: Sequence[ImagePayload]) -> list[dict[str, Any]]:
    """Interleave a label with each image so the model can refer to them by id."""
    blocks: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        blocks.append({"type": "text", "text": f"Image {index} (id: {image.id})"})
        blocks.append(image.to_content_block())
    return blocks


def _split_content(content: Sequence[dict[str, Any]]) -> tuple[str, list[str]]:
    """Flatten our provider-neutral blocks into (prompt_text, base64_images).

    Ollama's chat API takes one text body per message plus a sibling `images`
    array, rather than interleaved blocks.
    """
    text_parts: list[str] = []
    images: list[str] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            text_parts.append(str(block.get("text", "")))
        elif kind == "image":
            data = block.get("data")
            if data:
                images.append(data)
    return "\n\n".join(part for part in text_parts if part).strip(), images


def _extract_json(raw: str) -> str:
    """Pull a JSON object out of a model response.

    Schema-constrained decoding returns bare JSON, but a model that ignores the
    grammar (or a runtime that does not support it) may wrap it in prose or a
    markdown fence. Recover rather than fail the whole pipeline.
    """
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        return fenced.group("body").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


class LocalLLMClient:
    """Client for a locally-hosted inference server (Ollama-compatible)."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Any = None
        self._init_error: Optional[str] = None
        self._probe_error: Optional[str] = None
        # `None` means "never probed". Do not use 0.0 as the sentinel:
        # time.monotonic()'s epoch is platform-defined and starts near zero on
        # some systems, which would make the first probe look like a cached
        # failure from moments ago.
        self._probed_at: Optional[float] = None
        self._probe_ok: bool = False
        self._installed_models: list[str] = []
        self._lock = threading.Lock()

    # -- transport --------------------------------------------------------
    def _http(self) -> Any:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - environment dependent
                self._init_error = (
                    "The `httpx` package is not installed. "
                    "Run: pip install -r backend/requirements.txt"
                )
                raise LLMUnavailable(self._init_error) from exc
            # Vision models on CPU are slow; a generous read timeout beats
            # spurious failures mid-pipeline.
            self._client = httpx.Client(
                base_url=self._settings.llm_base_url,
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=self._settings.llm_timeout_seconds,
                    write=30.0,
                    pool=5.0,
                ),
            )
            return self._client

    # -- availability -----------------------------------------------------
    def _probe(self) -> bool:
        now = time.monotonic()
        if self._probed_at is not None and (now - self._probed_at) < _PROBE_TTL_SECONDS:
            return self._probe_ok

        self._probed_at = now
        if not self._settings.enable_llm:
            self._probe_ok = False
            self._probe_error = "Local LLM disabled via LSD_ENABLE_LLM=0"
            return False
        try:
            response = self._http().get("/api/tags")
            response.raise_for_status()
            payload = response.json()
        except LLMUnavailable:
            self._probe_ok = False
            self._probe_error = self._init_error
            return False
        except Exception as exc:  # noqa: BLE001 - any transport failure is "down"
            self._probe_ok = False
            self._probe_error = (
                f"Cannot reach the local model server at "
                f"{self._settings.llm_base_url} ({exc}). "
                "Start it with `ollama serve`, or run `make models` to install "
                "the models."
            )
            return False

        self._installed_models = [
            str(entry.get("name", "")) for entry in payload.get("models", [])
        ]
        missing = [
            name
            for name in {self._settings.text_model, self._settings.vision_model}
            if not self._has_model(name)
        ]
        if missing:
            self._probe_ok = False
            self._probe_error = (
                f"Model(s) not installed locally: {', '.join(sorted(missing))}. "
                f"Run `make models`, or `ollama pull {sorted(missing)[0]}`."
            )
            return False

        self._probe_ok = True
        self._probe_error = None
        return True

    def _has_model(self, name: str) -> bool:
        # Ollama reports "llava:7b"; accept a bare "llava" as a match for it.
        wanted = name.split(":")[0]
        return any(
            installed == name or installed.split(":")[0] == wanted
            for installed in self._installed_models
        )

    @property
    def available(self) -> bool:
        return self._probe()

    @property
    def unavailable_reason(self) -> str:
        if self._probe_ok:
            return ""
        return self._probe_error or self._init_error or "Local model unavailable"

    @property
    def text_model(self) -> str:
        return self._settings.text_model

    @property
    def vision_model(self) -> str:
        return self._settings.vision_model

    @property
    def installed_models(self) -> list[str]:
        self._probe()
        return list(self._installed_models)

    @property
    def base_url(self) -> str:
        return self._settings.llm_base_url

    # -- requests ---------------------------------------------------------
    def _chat(
        self,
        *,
        system: str,
        content: Sequence[dict[str, Any]],
        model: str,
        max_tokens: Optional[int],
        schema: Optional[dict] = None,
    ) -> str:
        if not self._probe():
            raise LLMNotRunning(self.unavailable_reason)

        prompt, images = _split_content(content)
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user_message["images"] = images

        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                user_message,
            ],
            "stream": False,
            # Keep the weights resident between agents — the pipeline makes
            # four sequential calls, and paying the load cost on each one
            # dominates total runtime on consumer hardware.
            "keep_alive": self._settings.llm_keep_alive,
            "options": {
                # Low temperature: these agents extract and reason, they do not
                # write creatively, and small local models drift badly when hot.
                "temperature": self._settings.llm_temperature,
                "num_ctx": self._settings.llm_context_tokens,
                "num_predict": max_tokens or self._settings.max_tokens,
            },
        }
        if schema is not None:
            # Schema-constrained decoding. This is the feature that makes the
            # structured agents dependable on a 7B local model.
            body["format"] = schema

        try:
            response = self._http().post("/api/chat", json=body)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            # A failure here means the pipeline degrades, so surface why.
            self._probe_ok = False
            self._probed_at = time.monotonic()
            self._probe_error = f"Local model request failed: {exc}"
            raise LLMUnavailable(self._probe_error) from exc

        if isinstance(payload, dict) and payload.get("error"):
            raise LLMUnavailable(f"Local model error: {payload['error']}")

        message = (payload or {}).get("message") or {}
        text = str(message.get("content", "")).strip()
        if not text:
            raise LLMUnavailable(f"Model '{model}' returned an empty response")
        return text

    # -- public helpers ---------------------------------------------------
    def structured(
        self,
        *,
        system: str,
        content: list[dict[str, Any]],
        output_model: type[T],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> T:
        """Run a schema-constrained request and return the validated object."""
        target_model = model or self._settings.text_model
        schema = output_model.model_json_schema()

        # Restating the schema in the system prompt is belt-and-braces: the
        # grammar already constrains the shape, but small models produce better
        # *values* when they can see the field descriptions.
        guided_system = (
            f"{system}\n\n"
            "Reply with a single JSON object matching this schema exactly. "
            "Output JSON only — no prose, no markdown fences.\n"
            f"{json.dumps(schema, indent=2)}"
        )

        raw = self._chat(
            system=guided_system,
            content=content,
            model=target_model,
            max_tokens=max_tokens,
            schema=schema,
        )

        try:
            return output_model.model_validate_json(_extract_json(raw))
        except (ValidationError, ValueError) as first_error:
            logger.warning(
                "%s failed schema validation on the first attempt (%s); retrying",
                output_model.__name__,
                str(first_error)[:200],
            )

        # One repair attempt: hand the model its own output plus the validation
        # error. Cheaper and more reliable than failing the whole agent.
        repair_content = list(content) + [
            {
                "type": "text",
                "text": (
                    "Your previous reply did not satisfy the schema:\n\n"
                    f"{raw[:2000]}\n\n"
                    "Reply again with only the corrected JSON object."
                ),
            }
        ]
        repaired = self._chat(
            system=guided_system,
            content=repair_content,
            model=target_model,
            max_tokens=max_tokens,
            schema=schema,
        )
        try:
            return output_model.model_validate_json(_extract_json(repaired))
        except (ValidationError, ValueError) as exc:
            raise LLMUnavailable(
                f"{output_model.__name__} response did not match the schema "
                f"after a repair attempt: {exc}"
            ) from exc

    def text(
        self,
        *,
        system: str,
        content: list[dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Run a plain-text request and return the response."""
        return self._chat(
            system=system,
            content=content,
            model=model or self._settings.text_model,
            max_tokens=max_tokens,
        )


_llm_client: Optional[LocalLLMClient] = None


def get_llm() -> LocalLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLLMClient()
    return _llm_client

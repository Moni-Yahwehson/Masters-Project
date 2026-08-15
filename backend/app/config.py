"""Runtime configuration for the LSD multi-agent diagnostic service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# backend/app/config.py -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "vision" / "cattle_lsd_detector.keras"
DEFAULT_KNOWLEDGE_DIR = BACKEND_ROOT / "app" / "rag" / "knowledge"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "vision" / "cow dataset"
DEFAULT_VISUAL_INDEX_PATH = PROJECT_ROOT / "vision" / "visual_index.npz"


ENV_FILE = BACKEND_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> int:
    """Load `KEY=VALUE` pairs from a .env file into the process environment.

    Hand-rolled rather than pulling in python-dotenv: the format we need is a
    dozen lines of parsing, and one fewer dependency matters for a project that
    has to be installable offline on consumer hardware.

    Real environment variables always win, so `LSD_TEXT_MODEL=x make backend`
    still overrides the file.
    """
    if not path.is_file():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        # Strip matching surrounding quotes, then drop trailing ` # comment`.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        os.environ[key] = value
        loaded += 1
    return loaded


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Environment-driven settings. Read once at import via `get_settings()`."""

    def __init__(self) -> None:
        # --- Local LLM (no hosted APIs) --------------------------------------
        # Inference runs on this machine via Ollama. Nothing leaves the host:
        # no cloud provider, no API key, no per-token cost. Point
        # LSD_LLM_BASE_URL at any server speaking the same /api/chat shape to
        # swap the runtime without touching the code.
        self.llm_base_url: str = os.getenv(
            "LSD_LLM_BASE_URL", "http://127.0.0.1:11434"
        ).rstrip("/")
        # Both roles default to ONE small multimodal model, which matters more
        # than raw model quality on consumer hardware: Ollama holds one model
        # in memory at a time, so a vision/text split makes it evict and reload
        # gigabytes of weights between agents — minutes of disk I/O per run.
        # gemma3:4b is ~3.3 GB, multimodal, and has a large context window.
        self.vision_model: str = os.getenv("LSD_VISION_MODEL", "gemma3:4b")
        self.text_model: str = os.getenv("LSD_TEXT_MODEL", "gemma3:4b")
        self.max_tokens: int = _env_int("LSD_LLM_MAX_TOKENS", 1536)
        self.llm_context_tokens: int = _env_int("LSD_LLM_CONTEXT_TOKENS", 8192)
        # Keep the model resident between agents. Without this Ollama unloads
        # after ~5 minutes idle and the next agent pays the full load cost again.
        self.llm_keep_alive: str = os.getenv("LSD_LLM_KEEP_ALIVE", "30m")
        # These agents extract and reason rather than write creatively, and
        # small local models drift badly when sampled hot.
        self.llm_temperature: float = _env_float("LSD_LLM_TEMPERATURE", 0.1)
        # Vision inference on CPU is slow; a generous read timeout beats
        # spurious mid-pipeline failures.
        self.llm_timeout_seconds: float = _env_float("LSD_LLM_TIMEOUT", 300.0)

        # --- CNN vision model -----------------------------------------------
        self.model_path: Path = Path(
            os.getenv("LSD_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        self.model_image_size: int = _env_int("LSD_MODEL_IMAGE_SIZE", 224)
        # Training used image_dataset_from_directory on `cow dataset/`, whose
        # alphabetically sorted class list is ["healthycows", "lumpycows"].
        # The head is a single sigmoid unit, so p = P(class 1) = P(lumpy).
        self.positive_class: str = os.getenv("LSD_POSITIVE_CLASS", "lumpycows")
        self.negative_class: str = os.getenv("LSD_NEGATIVE_CLASS", "healthycows")
        self.decision_threshold: float = _env_float("LSD_DECISION_THRESHOLD", 0.5)
        # Band around the threshold where the CNN is treated as inconclusive and
        # the LLM/RAG evidence is weighted more heavily by the synthesis agent.
        self.uncertainty_band: float = _env_float("LSD_UNCERTAINTY_BAND", 0.15)

        # --- Visual evidence --------------------------------------------------
        # Two deterministic explanations of the CNN's call, both derived from the
        # same forward pass the classifier already runs, so they cost no extra
        # model and no network:
        #
        #   * a Grad-CAM saliency overlay — which pixels moved the decision;
        #   * nearest labelled images from the training set in the classifier's
        #     own feature space — what the score is pattern-matching against.
        #
        # Both degrade to "absent" rather than failing the pipeline.
        self.enable_visual_evidence: bool = _env_bool("LSD_ENABLE_VISUAL_EVIDENCE", True)
        self.dataset_dir: Path = Path(
            os.getenv("LSD_DATASET_DIR", str(DEFAULT_DATASET_DIR))
        )
        self.visual_index_path: Path = Path(
            os.getenv("LSD_VISUAL_INDEX_PATH", str(DEFAULT_VISUAL_INDEX_PATH))
        )
        # Neighbours shown per diagnosis. 6 is two rows of three in the UI and
        # gives an odd-free majority that still reads at a glance.
        self.visual_neighbours: int = _env_int("LSD_VISUAL_NEIGHBOURS", 6)
        # Longest side of the rendered saliency image. 384 keeps the overlay
        # legible on a phone without bloating the JSON payload.
        self.saliency_render_size: int = _env_int("LSD_SALIENCY_RENDER_SIZE", 384)
        # Longest side of each stored precedent thumbnail.
        self.thumbnail_size: int = _env_int("LSD_THUMBNAIL_SIZE", 128)

        # --- RAG -------------------------------------------------------------
        self.knowledge_dir: Path = Path(
            os.getenv("LSD_KNOWLEDGE_DIR", str(DEFAULT_KNOWLEDGE_DIR))
        )
        self.retrieval_top_k: int = _env_int("LSD_RETRIEVAL_TOP_K", 6)
        self.retrieval_min_score: float = _env_float("LSD_RETRIEVAL_MIN_SCORE", 0.03)

        # --- Service ---------------------------------------------------------
        self.cors_origins: list[str] = [
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
            ).split(",")
            if origin.strip()
        ]
        self.max_images_per_request: int = _env_int("LSD_MAX_IMAGES", 6)
        self.max_image_bytes: int = _env_int("LSD_MAX_IMAGE_BYTES", 8 * 1024 * 1024)
        self.enable_llm: bool = _env_bool("LSD_ENABLE_LLM", True)
        self.session_ttl_seconds: int = _env_int("LSD_SESSION_TTL", 60 * 60 * 12)

    @property
    def models(self) -> set[str]:
        """The distinct local models this deployment needs pulled."""
        return {self.vision_model, self.text_model}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Populate the environment from backend/.env before reading any setting,
    # so the committed model configuration actually takes effect.
    load_env_file()
    return Settings()

"""FastAPI routers."""

from . import chat, diagnose, knowledge  # noqa: F401

__all__ = ["chat", "diagnose", "knowledge"]

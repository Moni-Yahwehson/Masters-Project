"""Individual agents in the LSD diagnostic workflow.

Each module exposes a single entry point and knows nothing about HTTP or the
other agents. `app.orchestrator` is responsible for sequencing them, handling
failures and assembling the report.
"""

from . import (  # noqa: F401 - re-exported for the orchestrator
    cnn_vision,
    knowledge_qa,
    lesion,
    retrieval,
    synthesis,
    triage,
    verification,
)

__all__ = [
    "cnn_vision",
    "knowledge_qa",
    "lesion",
    "retrieval",
    "synthesis",
    "triage",
    "verification",
]

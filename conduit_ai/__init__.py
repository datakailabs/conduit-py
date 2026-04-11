"""Conduit AI — Python SDK for the Conduit knowledge graph engine."""

from conduit_ai.client import ConduitClient
from conduit_ai.models import (
    AskResponse,
    ContextResponse,
    Source,
    RetrievalStats,
)

__all__ = [
    "ConduitClient",
    "AskResponse",
    "ContextResponse",
    "Source",
    "RetrievalStats",
    "LocalConduit",
]

__version__ = "0.2.1"


def __getattr__(name: str):
    if name == "LocalConduit":
        from conduit_ai.local import LocalConduit
        return LocalConduit
    raise AttributeError(f"module 'conduit_ai' has no attribute {name}")

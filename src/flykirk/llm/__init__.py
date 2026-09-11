"""Language-model layer: local inference plus brain-driven sampling controls."""

from .client import ChatClient, LLMError, OpenAICompatClient, ScriptedClient, TurnContext, make_client
from .sampling import DEFAULT_SAMPLING, SamplingParams, from_telemetry

__all__ = [
    "ChatClient",
    "OpenAICompatClient",
    "ScriptedClient",
    "TurnContext",
    "make_client",
    "LLMError",
    "SamplingParams",
    "DEFAULT_SAMPLING",
    "from_telemetry",
]

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


UUID4_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


class QueryRequest(BaseModel):
    """Incoming agent query payload."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        min_length=36,
        max_length=36,
        pattern=UUID4_PATTERN,
        description="Client session identifier in UUIDv4 string format.",
    )
    prompt: str = Field(
        min_length=1,
        description="User prompt to send to the agent.",
    )


class StreamPacket(BaseModel):
    """Internal telemetry packet emitted during streaming."""

    model_config = ConfigDict(extra="forbid")

    phase: str = Field(
        min_length=1,
        description="Current phase of the streaming workflow.",
    )
    message: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional human-readable status message for the current phase.",
    )
    text: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional text chunk associated with the current phase.",
    )

    @model_validator(mode="after")
    def validate_content_fields(self) -> "StreamPacket":
        if self.message is not None and self.text is not None:
            raise ValueError("Provide either 'message' or 'text', not both.")
        return self

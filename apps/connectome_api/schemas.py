from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiEnvelope(BaseModel):
    schema_version: str = "1.0"
    data: Any
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str
    schema_version: str = "1.0"


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None

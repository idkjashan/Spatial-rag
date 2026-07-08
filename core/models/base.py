# core/models/base.py
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum

class SchemaVersion(int, Enum):
    V1_0 = 100

class SpatialRAGBase(BaseModel):
    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique UUID")
    tenant_id: str = Field(default="default", description="Tenant isolation key")
    schema_version: SchemaVersion = Field(default=SchemaVersion.V1_0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    is_deleted: bool = Field(default=False)

    @field_validator('id', mode='before')
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {v}")
        return v

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
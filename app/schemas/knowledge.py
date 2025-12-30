import json
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator

MAX_CONTENT_LENGTH = 20000
MAX_METADATA_BYTES = 2048


class KnowledgeIngestRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, description="所属用户ID")
    echo_id: str = Field(..., min_length=1, max_length=64, description="数字分身ID (关键隔离字段)")
    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CONTENT_LENGTH,
        description="文档内容",
    )
    source_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="来源文件名",
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("content")
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @field_validator("metadata", mode="before")
    def validate_metadata(cls, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metadata must be a JSON object")
        serialized = json.dumps(value, ensure_ascii=False)
        if len(serialized.encode("utf-8")) > MAX_METADATA_BYTES:
            raise ValueError("metadata exceeds size limit")
        return value

class KnowledgeIngestResponse(BaseModel):
    status: str
    chunks_count: int
    message: str

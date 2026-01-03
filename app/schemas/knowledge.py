import json
from typing import Optional, Dict, Any, List

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
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
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

class KnowledgeDeleteRequest(BaseModel):
    knowledge_id: int = Field(..., description="业务系统中的知识ID (必须与训练时传入的一致)")
    echo_id: str = Field(..., description="数字分身ID (安全校验用)")
    user_id: str = Field(..., description="用户ID (安全校验用)")

class BatchKnowledgeDeleteRequest(BaseModel):
    # 接收一个列表
    items: List[KnowledgeDeleteRequest]

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class KnowledgeIngestRequest(BaseModel):
    user_id: str = Field(..., description="所属用户ID")
    echo_id: str = Field(..., description="数字分身ID (关键隔离字段)")
    content: str = Field(..., description="文档内容")
    source_name: str = Field(..., description="来源文件名")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class KnowledgeIngestResponse(BaseModel):
    status: str
    chunks_count: int
    message: str
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class KnowledgeIngestRequest(BaseModel):
    user_id: str = Field(..., description="用户ID，用于多租户隔离")
    content: str = Field(..., description="文档纯文本内容")
    source_name: str = Field(..., description="来源文件名")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="其他元数据")

class KnowledgeIngestResponse(BaseModel):
    status: str
    chunks_count: int
    message: str
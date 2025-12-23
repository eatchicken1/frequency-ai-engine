from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    """
    请求参数模型 (对应 Java 发送的 JSON)
    """
    session_id: str = Field(..., description="会话唯一标识")
    user_message: str = Field(..., description="用户发送的消息")
    echo_id: Optional[int] = Field(None, description="目标 Echo ID")
    # 用于 RAG 或个性化上下文的用户画像数据
    user_profile: Optional[Dict[str, Any]] = Field(None, description="用户画像数据")

class ChatResponse(BaseModel):
    """
    响应参数模型 (返回给 Java 的 JSON)
    """
    response_text: str = Field(..., description="AI 回复内容")
    is_finished: bool = Field(True, description="是否结束")
    # 扩展数据，可用于调试或返回消耗 token 数等
    data: Optional[Dict[str, Any]] = Field(None, description="扩展数据")
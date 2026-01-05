from pydantic import BaseModel, Field
from typing import Optional, Dict, Any,List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    echo_id: str = Field(..., description="数字分身ID")
    query: str = Field(..., description="用户提问")
    history: Optional[List[Message]] = Field(None, description="历史记录")
    echo_nickname: Optional[str] = "数字分身"
    echo_prompt: Optional[str] = ""  # 对应 personalityPrompt
    echo_tone: Optional[str] = ""  # 对应 voiceTone
    echo_tags: Optional[str] = ""  # 对应 tags

class ChatResponse(BaseModel):
    """
    响应参数模型 (返回给 Java 的 JSON)
    """
    response_text: str = Field(..., description="AI 回复内容")
    is_finished: bool = Field(True, description="是否结束")
    # 扩展数据，可用于调试或返回消耗 token 数等
    data: Optional[Dict[str, Any]] = Field(None, description="扩展数据")
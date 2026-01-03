from pydantic import BaseModel

class KnowledgeTrainRequest(BaseModel):
    knowledge_id: int
    user_id: str
    echo_id: str
    file_url: str
    file_type: str
    source_name: str

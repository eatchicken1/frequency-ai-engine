import dashscope
import redis
from http import HTTPStatus
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_redis import RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.schemas.knowledge import (
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
)

# ==============================================================================
# Redis Vector Config
# ==============================================================================
INDEX_NAME = "frequency_knowledge_idx"
KEY_PREFIX = "frequency:doc"


# ==============================================================================
# Redis Vector Capability Check
# ==============================================================================
def check_redis_vector_support(redis_url: str):
    r = redis.from_url(redis_url)
    try:
        r.execute_command("FT._LIST")
    except Exception as e:
        raise RuntimeError(
            "\n❌ Redis Vector Search NOT available.\n"
            "Please use Redis Stack:\n\n"
            "docker run -d -p 6379:6379 redis/redis-stack-server:latest\n\n"
            f"Original error: {e}"
        )


# ==============================================================================
# DashScope Embedding
# ==============================================================================
class FrequencyDashScopeEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-v1"):
        dashscope.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=texts,
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"{resp.code} - {resp.message}")

        return [
            item["embedding"]
            for item in sorted(
                resp.output["embeddings"],
                key=lambda x: x["text_index"],
            )
        ]

    def embed_query(self, text: str) -> List[float]:
        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=[text],
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"{resp.code} - {resp.message}")

        return resp.output["embeddings"][0]["embedding"]


# ==============================================================================
# Knowledge Engine
# ==============================================================================
class KnowledgeEngine:
    def __init__(self):
        # ⚠️ 注意：uvicorn --reload 下这里会执行两次
        check_redis_vector_support(settings.REDIS_URL)

        self.embeddings = FrequencyDashScopeEmbeddings(
            api_key=settings.OPENAI_API_KEY
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )

        # ✅ 关键修复点：embeddings（复数）
        self.vector_store = RedisVectorStore(
            redis_url=settings.REDIS_URL,
            index_name=INDEX_NAME,
            embeddings=self.embeddings,
            key_prefix=KEY_PREFIX,
        )

    # --------------------------------------------------------------------------
    async def ingest(
        self, request: KnowledgeIngestRequest
    ) -> KnowledgeIngestResponse:
        try:
            documents = self.text_splitter.create_documents(
                texts=[request.content],
                metadatas=[
                    {
                        "user_id": request.user_id,
                        "echo_id": request.echo_id,
                        "source": request.source_name,
                        **(request.metadata or {}),
                    }
                ],
            )

            if not documents:
                return KnowledgeIngestResponse(
                    status="warning",
                    chunks_count=0,
                    message="No content to ingest",
                )

            self.vector_store.add_documents(documents)

            return KnowledgeIngestResponse(
                status="success",
                chunks_count=len(documents),
                message=f"Ingested {len(documents)} chunks",
            )

        except Exception as e:
            return KnowledgeIngestResponse(
                status="error",
                chunks_count=0,
                message=str(e),
            )

    # --------------------------------------------------------------------------
    async def search(self, query: str, echo_id: str, k: int = 3):
        try:
            # [关键] Redis Stack 过滤语法：只检索当前 Echo 的记忆
            filter_expr = f'@echo_id:{{{echo_id}}}'

            return self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_expr  # 开启过滤
            )
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []


knowledge_engine = KnowledgeEngine()

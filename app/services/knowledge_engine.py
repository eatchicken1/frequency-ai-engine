import dashscope
import hashlib
import time
from http import HTTPStatus
from typing import List

import anyio
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections, utility, Collection

from app.core.config import settings
from app.core.logger import logger
from app.schemas.knowledge import (
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeDeleteRequest,
)

# ==============================================================================
# Milvus Vector Config
# ==============================================================================
COLLECTION_NAME = "frequency_knowledge"
DEDUPE_PREFIX = "frequency:ingest:dedupe"
DEDUPE_TTL_SECONDS = 60 * 60 * 24 * 7

# ==============================================================================
# Milvus Connection Check
# ==============================================================================
def check_milvus_connection() -> None:
    try:
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )
        utility.list_collections()
    except Exception as e:
        raise RuntimeError(
            "\n❌ Milvus NOT available.\n"
            "Please start Milvus:\n\n"
            "docker run -d -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.0\n\n"
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
def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _dedupe_key(echo_id: str, content_hash: str) -> str:
    return f"{DEDUPE_PREFIX}:{echo_id}:{content_hash}"


class KnowledgeEngine:
    def __init__(self):
        self.embeddings = FrequencyDashScopeEmbeddings(
            api_key=settings.OPENAI_API_KEY
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )

        self.vector_store = None
        self._dedupe_cache = {}

    def _ensure_vector_store(self) -> None:
        if self.vector_store is not None:
            return
        # ⚠️ 注意：uvicorn --reload 下这里会执行两次
        check_milvus_connection()
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={
                "host": settings.MILVUS_HOST,
                "port": settings.MILVUS_PORT,
            },
            auto_id=True,  # <--- 【关键修改】开启自动 ID 生成
            # drop_old=True, # ⚠️ 如果重启后报错 "Schema not match" (Schema不匹配)，请临时取消此行注释，运行一次以重建集合，然后再注释掉
        )

    # --------------------------------------------------------------------------
    async def ingest(
        self, request: KnowledgeIngestRequest
    ) -> KnowledgeIngestResponse:
        self._ensure_vector_store()
        content = request.content.strip()
        if not content:
            raise ValueError("content must not be blank")

        now = time.time()
        content_hash = _content_hash(content)
        dedupe_key = _dedupe_key(request.echo_id, content_hash)
        expired_keys = [
            key for key, expires_at in self._dedupe_cache.items()
            if expires_at <= now
        ]
        for key in expired_keys:
            self._dedupe_cache.pop(key, None)
        if dedupe_key in self._dedupe_cache:
            logger.info("Duplicate ingest skipped for echo_id=%s", request.echo_id)
            return KnowledgeIngestResponse(
                status="warning",
                chunks_count=0,
                message="Duplicate content skipped",
            )
        self._dedupe_cache[dedupe_key] = now + DEDUPE_TTL_SECONDS

        documents = self.text_splitter.create_documents(
            texts=[content],
            metadatas=[
                {
                    "user_id": request.user_id,
                    "echo_id": request.echo_id,
                    "source": request.source_name,
                    "content_hash": content_hash,
                    **(request.metadata or {}),
                }
            ],
        )

        if not documents:
            raise ValueError("No content to ingest")

        logger.info(
            "Ingesting knowledge: echo_id={}, user_id={}, chunks={}",
            request.echo_id,
            request.user_id,
            len(documents),
        )
        await anyio.to_thread.run_sync(self.vector_store.add_documents, documents)

        return KnowledgeIngestResponse(
            status="success",
            chunks_count=len(documents),
            message=f"Ingested {len(documents)} chunks",
        )

    async def delete(self, request: KnowledgeDeleteRequest):
        self._ensure_vector_store()

        # 构造 Milvus 的过滤表达式 (Expression)
        # LangChain 通常将元数据存在名为 "metadata" 的 JSON 字段中
        # 语法: metadata["key"] == value
        expr = f'metadata["knowledge_id"] == {request.knowledge_id} and metadata["echo_id"] == "{request.echo_id}"'

        logger.info(f"Deleting vectors with expr: {expr}")

        def _sync_delete():
            # 获取 Collection 对象
            # 优先尝试从 LangChain 实例获取，如果失败则手动加载
            if hasattr(self.vector_store, "col") and self.vector_store.col:
                col = self.vector_store.col
            else:
                col = Collection(COLLECTION_NAME)

            # 执行删除 (返回删除的条数在某些版本不准，主要看是否报错)
            col.delete(expr)
            return True

        # 放到线程池执行，避免阻塞
        await anyio.to_thread.run_sync(_sync_delete)

        return {"status": "success", "message": f"Deleted knowledge_id={request.knowledge_id}"}

    # --------------------------------------------------------------------------
    async def search(self, query: str, echo_id: str, k: int = 3):
        try:
            self._ensure_vector_store()
            # [关键] Milvus 过滤语法：只检索当前 Echo 的记忆
            filter_expr = f"metadata['echo_id'] == '{echo_id}'"

            return self.vector_store.similarity_search(
                query=query,
                k=k,
                expr=filter_expr
            )
        except Exception as e:
            logger.exception("Search error: {}", e)
            return []


knowledge_engine = KnowledgeEngine()

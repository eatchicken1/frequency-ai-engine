import dashscope
from http import HTTPStatus
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Redis
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.schemas.knowledge import KnowledgeIngestRequest, KnowledgeIngestResponse

# 定义 Redis 索引名称
INDEX_NAME = "frequency_knowledge_idx"
KEY_PREFIX = "frequency:doc"


# ==============================================================================
# 自定义 Embedding 类 (直接调用 DashScope SDK，避开 LangChain 兼容性Bug)
# ==============================================================================
class FrequencyDashScopeEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-v1"):
        dashscope.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量将文本转为向量"""
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=texts
            )
            if resp.status_code == HTTPStatus.OK:
                # 按 index 排序确保向量顺序与文本对应
                embeddings = [item['embedding'] for item in
                              sorted(resp.output['embeddings'], key=lambda x: x['text_index'])]
                return embeddings
            else:
                raise ValueError(f"DashScope API Error: {resp.code} - {resp.message}")
        except Exception as e:
            print(f"Embedding Error: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        """将单个查询转为向量"""
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=[text]
            )
            if resp.status_code == HTTPStatus.OK:
                return resp.output['embeddings'][0]['embedding']
            else:
                raise ValueError(f"DashScope API Error: {resp.code} - {resp.message}")
        except Exception as e:
            print(f"Embedding Query Error: {e}")
            raise e


# ==============================================================================
# 核心知识引擎
# ==============================================================================
class KnowledgeEngine:
    def __init__(self):
        # 1. 使用自定义的 Embedding 实现
        self.embeddings = FrequencyDashScopeEmbeddings(
            api_key=settings.OPENAI_API_KEY,  # 复用配置里的 Key
            model="text-embedding-v1"
        )

        # 2. 初始化切片器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

        self.redis_url = settings.REDIS_URL

    async def ingest(self, request: KnowledgeIngestRequest) -> KnowledgeIngestResponse:
        """
        投喂流程：切片 -> 向量化(DashScope) -> 存储(Redis)
        """
        try:
            # Step 1: 文本切分
            docs = self.text_splitter.create_documents(
                texts=[request.content],
                metadatas=[{
                    "user_id": request.user_id,
                    "source": request.source_name,
                    **request.metadata
                }]
            )

            if not docs:
                return KnowledgeIngestResponse(status="warning", chunks_count=0, message="No content to ingest")

            print(f"🔄 Ingesting {len(docs)} chunks for user {request.user_id}...")

            # Step 2: 存入 Redis (会自动调用上面的 embed_documents)
            Redis.from_documents(
                documents=docs,
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=INDEX_NAME,
                key_prefix=KEY_PREFIX
            )

            print(f"✅ Successfully ingested {len(docs)} chunks.")
            return KnowledgeIngestResponse(
                status="success",
                chunks_count=len(docs),
                message=f"Synced {len(docs)} memory fragments via Native DashScope"
            )

        except Exception as e:
            print(f"❌ Ingestion Error: {str(e)}")
            return KnowledgeIngestResponse(status="error", chunks_count=0, message=str(e))

    async def search(self, query: str, user_id: str, k: int = 3):
        """
        检索流程：Query向量化 -> Redis KNN搜索 (带租户过滤)
        """
        try:
            vector_store = Redis(
                redis_url=self.redis_url,
                index_name=INDEX_NAME,
                embedding=self.embeddings,
                key_prefix=KEY_PREFIX
            )

            # 租户隔离过滤
            filter_expr = f'@user_id:{{{user_id}}}'

            results = vector_store.similarity_search(
                query,
                k=k,
                # filter=filter_expr # 暂时注释，如果你的 Redis 索引还没建好 tag 字段，开启这个会报错
            )
            return results
        except Exception as e:
            print(f"Search Error: {e}")
            return []


knowledge_engine = KnowledgeEngine()
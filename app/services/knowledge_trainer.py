# app/services/knowledge_trainer.py
from app.services.file_loader import download_file_from_oss
from app.services.file_parsers import parse_file
from app.schemas.knowledge import KnowledgeIngestRequest
from app.services.knowledge_engine import knowledge_engine


async def train_from_oss(
    *,
    knowledge_id: int,
    user_id: str,
    echo_id: str,
    file_url: str,
    file_type: str,
    source_name: str,
):
    # 1️⃣ 下载 OSS 文件
    raw_bytes = await download_file_from_oss(
        region="cn-beijing",
        bucket="pig-frequency",
        object_key="knowledge/2026-01-01/xxxx.pdf",
    )

    # 2️⃣ 解析成纯文本
    content = parse_file(raw_bytes, file_type)

    # 3️⃣ 调用已有 ingest（chunk + embedding + milvus）
    ingest_req = KnowledgeIngestRequest(
        user_id=user_id,
        echo_id=echo_id,
        content=content,
        source_name=source_name,
        metadata={
            "knowledge_id": knowledge_id,
            "file_type": file_type,
        },
    )

    return await knowledge_engine.ingest(ingest_req)

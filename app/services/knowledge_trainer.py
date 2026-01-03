from urllib.parse import urlparse
from app.core.logger import logger
from app.services.file_loader import download_file_from_oss
from app.services.file_parsers import parse_file
from app.schemas.knowledge import KnowledgeIngestRequest
from app.services.knowledge_engine import knowledge_engine


def _parse_oss_url(url: str):
    """
    从 OSS URL 中解析 bucket 和 object_key
    示例 URL: https://pig-frequency.oss-cn-beijing.aliyuncs.com/knowledge/2024/test.pdf
    """
    parsed = urlparse(url)

    # 1. 域名通常是 <bucket>.<endpoint>
    # 例如: pig-frequency.oss-cn-beijing.aliyuncs.com
    domain_parts = parsed.netloc.split('.', 1)
    if len(domain_parts) < 2:
        raise ValueError(f"Invalid OSS URL format: {url}")

    bucket = domain_parts[0]
    # endpoint = domain_parts[1] # 如果需要动态获取 endpoint 可以用这个

    # 2. Path 去掉开头的 / 就是 object_key
    object_key = parsed.path.lstrip('/')

    return bucket, object_key


async def train_from_oss(
        *,
        knowledge_id: int,
        user_id: str,
        echo_id: str,
        file_url: str,
        file_type: str,
        source_name: str,
):
    logger.info(f"Start training from OSS: url={file_url}")

    # 1️⃣ 解析 URL (替换原本的硬编码逻辑)
    try:
        bucket, object_key = _parse_oss_url(file_url)
    except Exception as e:
        logger.error(f"Failed to parse OSS URL: {e}")
        raise ValueError(f"无效的 OSS 链接: {file_url}")

    # 2️⃣ 下载 OSS 文件
    # 注意：这里 region 暂时硬编码为 "cn-beijing"，
    # 如果你的 bucket 在不同区域，建议将 region 也放入 .env 配置或从 URL endpoint 中解析
    raw_bytes = await download_file_from_oss(
        region="cn-beijing",
        bucket=bucket,
        object_key=object_key,
    )

    # 3️⃣ 解析成纯文本
    content = parse_file(raw_bytes, file_type)

    # 4️⃣ 调用已有 ingest（chunk + embedding + milvus）
    ingest_req = KnowledgeIngestRequest(
        user_id=user_id,
        echo_id=echo_id,
        content=content,
        source_name=source_name,
        metadata={
            "knowledge_id": knowledge_id,
            "file_type": file_type,
            "original_url": file_url
        },
    )

    return await knowledge_engine.ingest(ingest_req)
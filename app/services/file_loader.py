import anyio
import alibabacloud_oss_v2 as oss
import logger

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
CHUNK_SIZE = 256 * 1024  # 256KB


def _download_from_oss_sync(
    *,
    region: str,
    bucket: str,
    object_key: str,
    endpoint: str | None = None,
) -> bytes:
    """
    使用阿里云 OSS 官方 SDK 同步下载（内部函数）
    """
    logger.info("Downloading from OSS: %s/%s", bucket, object_key)
    # 1️⃣ 凭证（推荐用环境变量）
    credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()

    # 2️⃣ SDK 配置
    cfg = oss.config.load_default()
    cfg.credentials_provider = credentials_provider
    cfg.region = region
    if endpoint:
        cfg.endpoint = endpoint

    # 3️⃣ 客户端
    client = oss.Client(cfg)

    # 4️⃣ GetObject
    result = client.get_object(
        oss.GetObjectRequest(
            bucket=bucket,
            key=object_key,
        )
    )

    # 5️⃣ 分块读取 + 限制大小
    total = 0
    chunks: list[bytes] = []

    with result.body as body_stream:
        for chunk in body_stream.iter_bytes(block_size=CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                raise RuntimeError("File too large")

            chunks.append(chunk)

    if not chunks:
        raise RuntimeError("Empty file")

    return b"".join(chunks)


async def download_file_from_oss(
    *,
    region: str,
    bucket: str,
    object_key: str,
    endpoint: str | None = None,
) -> bytes:
    """
    FastAPI 可用的异步包装
    """
    return await anyio.to_thread.run_sync(
        _download_from_oss_sync,
        region=region,
        bucket=bucket,
        object_key=object_key,
        endpoint=endpoint,
    )

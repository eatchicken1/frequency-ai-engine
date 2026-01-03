import functools
import anyio
import alibabacloud_oss_v2 as oss
from app.core.logger import logger
from app.core.config import settings

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
    logger.info(f"Downloading from OSS: region={region}, bucket={bucket}, object_key={object_key}")

    if not settings.OSS_ACCESS_KEY_ID or not settings.OSS_ACCESS_KEY_SECRET:
        error_msg = "OSS Access Key 未配置，请在 .env 文件中填写 OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET"
        logger.error(error_msg)
        raise ValueError(error_msg)

    credentials_provider = oss.credentials.StaticCredentialsProvider(
        access_key_id=settings.OSS_ACCESS_KEY_ID,
        access_key_secret=settings.OSS_ACCESS_KEY_SECRET
    )

    cfg = oss.config.load_default()
    cfg.credentials_provider = credentials_provider
    cfg.region = region
    if endpoint:
        cfg.endpoint = endpoint

    client = oss.Client(cfg)

    try:
        result = client.get_object(
            oss.GetObjectRequest(
                bucket=bucket,
                key=object_key,
            )
        )
    except Exception as e:
        logger.error(f"OSS download failed: {e}")
        raise e

    total = 0
    chunks: list[bytes] = []

    with result.body as body_stream:
        for chunk in body_stream.iter_bytes(block_size=CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                raise RuntimeError(f"File too large (exceeds {MAX_FILE_SIZE} bytes)")

            chunks.append(chunk)

    if not chunks:
        logger.warning(f"Downloaded empty file: {object_key}")
        return b""

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
    # --- 2. 核心修改开始 ---
    # 使用 functools.partial 将关键字参数绑定到函数上
    # anyio.to_thread.run_sync(func, *args) 不支持 kwargs，所以必须这样写
    func = functools.partial(
        _download_from_oss_sync,
        region=region,
        bucket=bucket,
        object_key=object_key,
        endpoint=endpoint,
    )
    return await anyio.to_thread.run_sync(func)
    # --- 核心修改结束 ---
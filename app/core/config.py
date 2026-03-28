from typing import Optional # 记得导入 Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    系统配置加载类
    自动读取环境变量或 .env 文件
    """
    PROJECT_NAME: str = "Frequency Agent Runtime"
    API_V1_STR: str = "/api/v1"

    # 基础配置
    PORT: int = 8000
    ENV_MODE: str = "dev"
    LOG_LEVEL: str = "INFO"  # 上一步添加的日志级别

    # --- 新增：阿里云 OSS 配置 (解决报错的关键) ---
    # 定义这两个字段后，Pydantic 就不会报错了，而且代码里可以直接用 settings.OSS_ACCESS_KEY_ID
    OSS_ACCESS_KEY_ID: Optional[str] = None
    OSS_ACCESS_KEY_SECRET: Optional[str] = None

    # 外部服务地址
    PIG_API_URL: str = "http://localhost:9999"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # AI 模型配置
    OPENAI_API_KEY: str = "sk-..."
    OPENAI_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # --- Runtime 服务间鉴权配置 ---
    # Java(pig-ai-agent) 与 Python runtime 间共享密钥（HMAC-SHA256）
    RUNTIME_SHARED_SECRET: str = ""
    # 允许调用 runtime 的 client_id 白名单，逗号分隔
    RUNTIME_ALLOWED_CLIENTS: str = "pig-ai-agent"
    # 签名时间窗（秒）
    RUNTIME_CLOCK_SKEW_SECONDS: int = 120
    # 仅开发调试时可开启，允许缺少签名
    RUNTIME_ALLOW_INSECURE: bool = False
    # nonce 防重放存储（redis / memory）
    RUNTIME_NONCE_STORE: str = "redis"
    # nonce redis key 前缀
    RUNTIME_NONCE_REDIS_PREFIX: str = "freq:runtime:nonce:"
    # Redis URL（建议通过环境变量注入）
    REDIS_URL: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

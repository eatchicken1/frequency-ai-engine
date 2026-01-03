from typing import Optional # 记得导入 Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    系统配置加载类
    自动读取环境变量或 .env 文件
    """
    PROJECT_NAME: str = "Frequency AI Engine"
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

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
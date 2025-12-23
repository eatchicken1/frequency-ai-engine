from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    系统配置加载类
    自动读取环境变量或 .env 文件
    """
    PROJECT_NAME: str = "Frequency AI Engine"
    API_V1_STR: str = "/api/v1"

    # 基础配置 (从 .env 读取，若无则使用默认值)
    PORT: int = 8000
    ENV_MODE: str = "dev"

    # 外部服务地址
    PIG_API_URL: str = "http://localhost:9999"
    REDIS_URL: str = "redis://localhost:6379/0"

    # AI 模型配置
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = ""

    class Config:
        # 指定配置文件路径
        env_file = ".env"
        # 允许大小写匹配
        case_sensitive = True


# 单例导出，供其他模块导入使用
settings = Settings()
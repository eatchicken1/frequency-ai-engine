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
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # AI 模型配置
    OPENAI_API_KEY: str = "sk-a7aa6ef6ca5d4010a23cbfafb48a2978"
    OPENAI_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    class Config:
        # 指定配置文件路径
        env_file = ".env"
        # 允许大小写匹配
        case_sensitive = True


# 单例导出，供其他模块导入使用
settings = Settings()

from langchain_openai import ChatOpenAI
from app.core.config import settings

def get_llm(temperature: float = 0.7):
    """
    获取 LangChain 的 LLM 实例
    支持任何兼容 OpenAI 协议的模型 (DeepSeek, Moonshot, Qwen 等)
    """
    # 如果没有配置 Key，这里会报错，提醒开发者配置 .env
    if not settings.OPENAI_API_KEY:
        raise ValueError("请在 .env 文件中配置 OPENAI_API_KEY")

    return ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        model_name="qwen-plus", # 很多国产模型兼容接口时忽略此参数，或填具体模型名如 "deepseek-chat"
        temperature=temperature,    # 0.7 比较适合闲聊，更有创造力
        streaming=True              # 准备支持流式输出
    )
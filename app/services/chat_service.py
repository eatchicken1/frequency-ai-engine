from typing import List  # <--- 1. 新增导入
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage  # <--- 2. 新增 BaseMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.core.logger import logger
from app.services.knowledge_engine import knowledge_engine
from app.schemas.chat import ChatRequest


async def chat_stream_generator(request: ChatRequest):
    """
    RAG 对话流式生成器
    """
    try:
        # 1. 检索相关知识 (RAG)
        docs = await knowledge_engine.search(request.query, request.echo_id, limit=3)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        logger.info(f"Retrieved {len(docs)} docs for echo_id={request.echo_id}")

        # 2. 构建 Prompt
        system_prompt = f"""
        你是一个数字分身。请基于以下[参考知识]回答用户的问题。
        如果参考知识中没有答案，请根据你的常识回答，但要保持语气符合人设。

        [参考知识]:
        {context_text}
        """

        # [核心修改]: 显式声明列表类型为 List[BaseMessage]，解决类型推断错误
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

        # 确保 history 不为 None
        history = request.history or []

        # 添加历史记录
        for msg in history[-4:]:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role in ['assistant', 'ai']:
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=request.query))

        # 3. 初始化 LLM
        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            model="qwen-plus",
            temperature=0.7,
            streaming=True
        )

        # 4. 流式调用
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        yield f"Error: {str(e)}"
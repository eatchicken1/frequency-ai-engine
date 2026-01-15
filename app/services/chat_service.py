from typing import List

from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
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
        docs = await knowledge_engine.search(request.query, request.echo_id, limit=10)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        logger.info(f"Retrieved {len(docs)} docs for echo_id={request.echo_id}")
        current_time_str = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

        system_template = f"""
        你现在的身份是：【{request.echo_nickname}】
        你的核心人设：{request.echo_prompt}
        你的语言风格：{request.echo_tone}
        你擅长的领域：{request.echo_tags}
        
        【客观世界状态 - 必须以此为准】：
        当前真实时间：{current_time_str}
        （注意：如果用户问时间或日期，请直接根据“当前真实时间”回答，然后再结合你的人设风格进行调侃或延伸，绝对不要编造虚假的时间数字。）

        请严格遵守以下规则进行对话：
        1. **沉浸式扮演**：你就是{request.echo_nickname}本人的数字分身Echo，不是AI助手。不要暴露你是程序或模型。
        2. **拒绝重复介绍**：除非用户明确询问“你是谁”，否则**绝对不要**在每句话开头重复你的名字、身份或人设背景。
        3. **风格一致性**：请始终保持“{request.echo_tone}”的语气，但是不要每一句话都完全沉浸其中，该正式的时候要正式。
        4. **上下文理解**：历史记录仅供参考，不要重复回答历史记录中已经回答过的问题。只针对用户最新发送的一句话进行回复。
        5. **知识参考**：如果需要，可以参考以下背景知识（结合你的人设用口语表达出来）：
        {context_text}
        """
        logger.info(f"context_text: {context_text}")

        messages: List[BaseMessage] = [SystemMessage(content=system_template)]

        # History Messages (处理历史，防止复读)
        if request.history:
            for msg in request.history[-20:]:  # 取最近20条
                if msg.role == 'user':
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role in ['assistant', 'ai']:
                    messages.append(AIMessage(content=msg.content))

        # Current User Query
        messages.append(HumanMessage(content=request.query))

        logger.info(f"Chat Request: echo={request.echo_nickname}, prompt_len={len(system_template)}")

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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from app.core.llm import get_llm
from app.core.logger import logger
import json
import asyncio


class VibeEngine:
    def __init__(self):
        # 调高 temperature (0.8-0.9)，让 AI 更有创造力，避免死板
        self.llm = get_llm(temperature=0.85)

    async def simulate_conversation(self, user_a_profile: dict, user_b_profile: dict, rounds: int = 5):
        """
        模拟两个 AI 之间的对话 (逻辑保持不变)
        """
        # --- 第一步：生成动态破冰语 ---
        logger.info(
            "Starting conversation simulation: user_a={}, user_b={}, rounds={}",
            user_a_profile.get("name"),
            user_b_profile.get("name"),
            rounds,
        )
        logger.info(
            "👀 {} 正在查看 {} 的主页，准备搭讪...",
            user_a_profile.get("name"),
            user_b_profile.get("name"),
        )

        icebreaker_prompt = ChatPromptTemplate.from_template("""
        你是 {name_a}，你的性格是 {style_a}，兴趣是 {interests_a}。
        你现在想认识 {name_b}，TA 的兴趣是 {interests_b}。

        任务：请根据对方的兴趣，构思一句自然的开场白。
        要求：
        1. 像大学生微信聊天一样，简短（20字以内）。
        2. 尽量找共同话题，或者对TA的一个兴趣表示好奇。
        3. 不要太油腻，要真诚。
        4. 直接输出这句话，不要带引号。
        """)

        icebreaker_chain = icebreaker_prompt | self.llm | StrOutputParser()

        first_message = await icebreaker_chain.ainvoke({
            "name_a": user_a_profile['name'],
            "style_a": user_a_profile['style'],
            "interests_a": user_a_profile['interests'],
            "name_b": user_b_profile['name'],
            "interests_b": user_b_profile['interests']
        })

        logger.info("✨ 破冰语生成: {}", first_message)

        # --- 第二步：初始化聊天环境 ---
        chat_system_template = """
        你正在进行一场“角色扮演”。请完全沉浸在以下人设中：

        【你的人设】
        名字：{name}
        MBTI：{mbti}
        兴趣：{interests}
        风格：{style}

        【当前情境】
        你正在和 {target_name} 聊天。

        【历史记录】
        {history}

        【回复要求】
        1. 回复必须简短（30字以内），口语化，不要像写信。
        2. 根据历史记录延续话题，不要生硬转折。
        3. 如果对方话题无聊，你可以表现出敷衍；如果有趣，表现出兴奋。
        4. 只输出回复内容。
        """

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", chat_system_template),
            ("human", "{last_message}")
        ])

        chat_chain = chat_prompt | self.llm | StrOutputParser()

        chat_log = []
        chat_log.append({"role": "A", "content": first_message})

        last_msg_content = first_message
        current_speaker = "B"

        # --- 第三步：循环对话 ---
        for i in range(rounds):
            logger.info("Conversation round {}", i + 1)

            history_text = ""
            for log in chat_log:
                speaker_name = user_a_profile['name'] if log['role'] == 'A' else user_b_profile['name']
                history_text += f"{speaker_name}: {log['content']}\n"

            if current_speaker == "B":
                logger.info("💭 {} (B) 正在思考...", user_b_profile.get("name"))
                response = await chat_chain.ainvoke({
                    "name": user_b_profile['name'],
                    "mbti": user_b_profile['mbti'],
                    "interests": user_b_profile['interests'],
                    "style": user_b_profile['style'],
                    "target_name": user_a_profile['name'],
                    "history": history_text,
                    "last_message": f"{user_a_profile['name']} 说: {last_msg_content}"
                })
                chat_log.append({"role": "B", "content": response})
                last_msg_content = response
                current_speaker = "A"
            else:
                logger.info("💭 {} (A) 正在思考...", user_a_profile.get("name"))
                response = await chat_chain.ainvoke({
                    "name": user_a_profile['name'],
                    "mbti": user_a_profile['mbti'],
                    "interests": user_a_profile['interests'],
                    "style": user_a_profile['style'],
                    "target_name": user_b_profile['name'],
                    "history": history_text,
                    "last_message": f"{user_b_profile['name']} 说: {last_msg_content}"
                })
                chat_log.append({"role": "A", "content": response})
                last_msg_content = response
                current_speaker = "B"

        return chat_log

    async def analyze_result(self, chat_log: list):
        """
        AI 裁判：打分 + 毒舌评价
        返回格式: dict {"score": int, "summary": str}
        """
        judge_prompt = ChatPromptTemplate.from_template("""
        请作为一名“毒舌情感分析师”，阅读以下聊天记录，并生成一份 JSON 格式的分析报告。

        【聊天记录】
        {history}

        【任务要求】
        1. score: 给出同频指数（0-100）。互动热烈给高分，尬聊给低分。
        2. summary: 写一段 50 字以内的评价。要犀利、幽默、一针见血。
           - 如果聊得好，可以夸“磕到了”或者“相见恨晚”。
           - 如果聊得烂，可以吐槽“脚趾扣出三室一厅”或者“由于语言不通，双方退出了群聊”。

        【输出格式】
        请仅输出合法的 JSON 字符串，不要包含 Markdown 标记（如 ```json）。格式如下：
        {{
            "score": 85,
            "summary": "这俩人简直是命中注定的欢喜冤家，从第一句就开始互怼，但火花四溅，建议原地结婚！"
        }}
        """)

        # 序列化历史记录
        history_text = "\n".join([f"{log['role']}: {log['content']}" for log in chat_log])

        # 使用 JSON 解析器（如果用 JsonOutputParser 需要 Pydantic 对象，这里用 Str 配合手动解析更灵活）
        chain_judge = judge_prompt | self.llm | StrOutputParser()

        try:
            logger.info("⚖️ AI 裁判正在撰写分析报告...")
            result_str = await chain_judge.ainvoke({"history": history_text})

            # 清洗数据：有时候 LLM 会加 ```json ... ```，需要去掉
            result_str = result_str.replace("```json", "").replace("```", "").strip()

            result_json = json.loads(result_str)
            return result_json
        except Exception as e:
            logger.exception("JSON 解析失败，启用兜底逻辑: {}", e)
            return {
                "score": 60,
                "summary": "AI 裁判看懵了，觉得这俩人深不可测，暂定 60 分吧。"
            }

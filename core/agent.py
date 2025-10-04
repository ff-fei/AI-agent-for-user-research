from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from core.prompts import create_chat_prompt
from config.config import Config
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RestaurantWorkerAgent:
    """餐饮服务行业从业者AI Agent"""

    def __init__(self):
        """初始化Agent"""
        try:
            # 初始化OpenAI Chat模型
            # self.llm = ChatOpenAI(
            #     openai_api_key=Config.OPENAI_API_KEY,
            #     model=Config.OPENAI_MODEL,
            #     temperature=Config.TEMPERATURE
            # )

            # 初始化DeepSeek模型
            self.llm = ChatOpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                model_name=Config.DEEPSEEK_MODEL,
                temperature=Config.TEMPERATURE,
                base_url="https://api.deepseek.com/v1"
            )

            # 创建提示词模板
            self.prompt = create_chat_prompt()

            # 初始化对话记忆（保留最近50条消息）
            self.memory = ConversationBufferWindowMemory(
                k=Config.MAX_MEMORY_MESSAGES,
                return_messages=True,
                memory_key="chat_history"
            )

            # 创建对话链
            self.chain = self.prompt | self.llm

            logger.info("餐饮服务Agent初始化成功")

        except Exception as e:
            logger.error(f"Agent初始化失败: {e}")
            raise

    def chat(self, user_input: str) -> str:
        """
        处理用户输入并返回AI回复

        Args:
            user_input (str): 用户输入的消息

        Returns:
            str: AI的回复消息
        """
        try:
            # 获取对话历史
            chat_history = self.memory.chat_memory.messages

            # 调用AI模型生成回复
            response = self.chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            # 提取回复内容
            ai_response = response.content

            # 保存对话到记忆
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(ai_response)

            logger.info(f"用户输入: {user_input[:50]}...")
            logger.info(f"AI回复: {ai_response[:50]}...")

            return ai_response

        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            return f"抱歉，我现在有点忙，请稍后再试。错误信息: {str(e)}"

    def get_conversation_history(self) -> list:
        """
        获取对话历史记录

        Returns:
            list: 对话历史列表，每个元素包含角色和消息内容
        """
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history

    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()
        logger.info("对话记忆已清空")

    def get_memory_length(self) -> int:
        """获取当前记忆中的消息数量"""
        return len(self.memory.chat_memory.messages)
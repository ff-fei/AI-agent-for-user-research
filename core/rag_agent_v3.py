"""
RAG Agent V3 - 调试增强版
每次对话都会输出检索到的知识库内容，方便确认模型是否使用。
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from core.prompts import create_chat_prompt
from core.knowledge_loader import KnowledgeLoader
from config.config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RAGRestaurantAgentV3:
    """基于Chroma的餐饮服务AI Agent (调试增强版)"""

    def __init__(self, knowledge_loader: KnowledgeLoader):
        try:
            # 初始化 LLM
            # self.llm = ChatOpenAI(
            #     openai_api_key=Config.OPENAI_API_KEY,
            #     model=Config.OPENAI_MODEL,
            #     temperature=Config.TEMPERATURE
            # )

            self.llm = ChatOpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                model_name=Config.DEEPSEEK_MODEL,
                temperature=Config.TEMPERATURE,
                base_url="https://api.deepseek.com/v1"  # DeepSeek API基础地址
            )

            self.knowledge_loader = knowledge_loader
            self.prompt = create_chat_prompt()

            # 初始化对话记忆
            self.memory = ConversationBufferWindowMemory(
                k=Config.MAX_MEMORY_MESSAGES,
                return_messages=True,
                memory_key="chat_history"
            )

            # chain = prompt | llm
            self.chain = self.prompt | self.llm

            logger.info("RAG Agent V3 初始化成功 (调试增强版)")

        except Exception as e:
            logger.error(f"RAG Agent V3 初始化失败: {e}")
            raise

    def chat(self, user_input: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        处理用户输入并返回 AI 回复

        Args:
            user_input: 用户输入
            k: 检索条数（如果不指定，默认取 10）
        """
        try:
            k = k or 20

            # 1. 检索知识库
            relevant_docs = self.knowledge_loader.search(user_input, k=k)

            logger.info(f"[DEBUG] 检索到 {len(relevant_docs)} 条知识库内容")
            for i, doc in enumerate(relevant_docs[:5], 1):  # 只预览前5条
                preview = doc.get("content", "")[:300].replace("\n", " ")
                logger.info(f"[DEBUG] 文档 {i}: {preview}")

            # 2. 拼接 knowledge_context（调试：不做过滤，完整拼接）
            knowledge_context = self._build_knowledge_context(relevant_docs)
            logger.info(f"[DEBUG] 拼接的 knowledge_context 内容:\n{knowledge_context[:1000]}")

            # 3. 获取对话历史
            chat_history = self.memory.chat_memory.messages

            # 4. 调用 LLM
            response = self.chain.invoke({
                "input": user_input,
                "chat_history": chat_history,
                "knowledge_context": knowledge_context
            })

            ai_response = response.content

            # 5. 更新记忆
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(ai_response)

            return {
                "response": ai_response,
                "retrieved_count": len(relevant_docs),
                "knowledge_used": len(relevant_docs),
                "retrieved_docs": relevant_docs,
                "knowledge_context": knowledge_context,
                "memory_count": self.get_memory_length()
            }

        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            return {
                "response": f"抱歉，出错了：{e}",
                "retrieved_count": 0,
                "knowledge_used": 0,
                "retrieved_docs": [],
                "knowledge_context": "",
                "memory_count": self.get_memory_length()
            }

    def _build_knowledge_context(self, relevant_docs: List[Dict]) -> str:
        """
        构建知识上下文字符串（完整拼接，不筛选）
        """
        if not relevant_docs:
            return "暂无相关经验。"

        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            content = doc.get("content", "").strip()
            sim = doc.get("similarity_score", None)
            header = f"【参考 {i}"
            if sim is not None:
                try:
                    header += f" | 相似度: {float(sim):.2f}"
                except:
                    header += f" | 相似度: {sim}"
            header += "】"
            context_parts.append(f"{header}\n{content}")

        return "\n\n".join(context_parts)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history

    def clear_memory(self):
        """清空记忆"""
        self.memory.clear()
        logger.info("对话记忆已清空")

    def get_memory_length(self) -> int:
        """获取记忆条数"""
        return len(self.memory.chat_memory.messages)

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """返回知识库统计"""
        return self.knowledge_loader.get_stats()

    def search_knowledge(self, query: str, k: int = 5) -> List[Dict]:
        """搜索知识库（便于调试）"""
        return self.knowledge_loader.search(query, k=k)

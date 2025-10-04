#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
餐饮服务AI Agent - Version 4 完整后端
基于Chroma向量数据库和text2vec-base-chinese的RAG系统
"""

import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_agent_v3 import RAGRestaurantAgentV3
from core.knowledge_loader import KnowledgeLoader
from config.config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 全局变量
knowledge_loader = None
agent = None


def initialize_system():
    """初始化系统"""
    global knowledge_loader, agent

    try:
        logger.info("=" * 60)
        logger.info("开始初始化餐饮服务AI Agent V4系统")
        logger.info("=" * 60)

        # 1. 创建知识库加载器
        logger.info("步骤1: 创建知识库加载器...")
        knowledge_loader = KnowledgeLoader(persist_directory="./chroma_db")
        logger.info("✓ 知识库加载器创建成功")

        # 2. 检查知识库文件
        knowledge_file = r"E:\项目\AHFE\AI Agent\data\餐饮服务员RAG知识库.docx"
        logger.info(f"步骤2: 检查知识库文件: {knowledge_file}")

        if not os.path.exists(knowledge_file):
            logger.error(f"✗ 找不到知识库文件: {knowledge_file}")
            raise FileNotFoundError(f"知识库文件不存在: {knowledge_file}")

        file_size = os.path.getsize(knowledge_file) / 1024  # KB
        logger.info(f"✓ 知识库文件存在 (大小: {file_size:.2f} KB)")

        # 3. 处理知识库文件
        logger.info("步骤3: 加载和处理知识库...")
        knowledge_loader.process_knowledge_file(knowledge_file, force_rebuild=False)
        logger.info("✓ 知识库处理完成")

        # 4. 初始化RAG Agent
        logger.info("步骤4: 初始化RAG Agent...")
        agent = RAGRestaurantAgentV3(knowledge_loader)
        logger.info("✓ RAG Agent初始化成功")

        # 5. 获取统计信息
        stats = knowledge_loader.get_stats()
        logger.info("=" * 60)
        logger.info("系统初始化完成！")
        logger.info(f"知识库状态: {stats.get('status', '未知')}")
        logger.info(f"文档总数: {stats.get('total_documents', 0)}")
        logger.info(f"Embedding模型: {stats.get('embedding_model', '未知')}")
        logger.info(f"存储路径: {stats.get('persist_directory', '未知')}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ 系统初始化失败: {e}", exc_info=True)
        raise


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    处理聊天请求

    POST /api/chat
    Body: {"message": "用户消息"}

    Returns:
    {
        "success": true,
        "response": "AI回复",
        "memory_count": 2,
        "knowledge_used": 3,
        "rag_enhanced": true
    }
    """
    try:
        # 检查Agent是否初始化
        if agent is None:
            logger.error("Agent未初始化")
            return jsonify({
                "success": False,
                "error": "Agent未正确初始化，请检查配置"
            }), 500

        # 获取请求数据
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "请求格式错误，需要提供message字段"
            }), 400

        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                "success": False,
                "error": "消息内容不能为空"
            }), 400

        logger.info(f"收到聊天请求: {user_message[:50]}...")

        # 调用Agent处理消息
        result = agent.chat(user_message)

        logger.info(f"回复生成成功，使用了 {result.get('knowledge_used', 0)} 条知识")

        return jsonify({
            "success": True,
            "response": result['response'],
            "memory_count": result.get('memory_count', 0),
            "knowledge_used": result.get('knowledge_used', 0),
            "rag_enhanced": True
        })

    except Exception as e:
        logger.error(f"聊天处理失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }), 500


@app.route('/api/search_knowledge', methods=['POST'])
def search_knowledge():
    """
    搜索知识库

    POST /api/search_knowledge
    Body: {"query": "搜索关键词", "limit": 5}

    Returns:
    {
        "success": true,
        "results": [
            {
                "content": "文档内容",
                "full_content": "完整内容",
                "similarity_score": 0.85,
                "metadata": {}
            }
        ],
        "total": 5
    }
    """
    try:
        if knowledge_loader is None:
            return jsonify({
                "success": False,
                "error": "知识库未初始化"
            }), 500

        data = request.get_json()
        query = data.get('query', '').strip()
        limit = data.get('limit', 5)

        if not query:
            return jsonify({
                "success": False,
                "error": "搜索关键词不能为空"
            }), 400

        logger.info(f"搜索知识库: {query}, limit={limit}")

        # 搜索知识库
        results = knowledge_loader.search(query, k=limit)

        # 格式化结果
        formatted_results = []
        for result in results:
            content = result['content']
            formatted_results.append({
                "content": content[:300] + ('...' if len(content) > 300 else ''),
                "full_content": content,
                "similarity_score": result['similarity_score'],
                "metadata": result.get('metadata', {})
            })

        logger.info(f"搜索完成，找到 {len(results)} 条结果")

        return jsonify({
            "success": True,
            "results": formatted_results,
            "total": len(results)
        })

    except Exception as e:
        logger.error(f"知识搜索失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"搜索失败: {str(e)}"
        }), 500


@app.route('/api/knowledge_stats', methods=['GET'])
def get_knowledge_stats():
    """
    获取知识库统计信息

    GET /api/knowledge_stats

    Returns:
    {
        "success": true,
        "stats": {
            "status": "已加载",
            "total_documents": 150,
            "embedding_model": "text2vec-base-chinese",
            "persist_directory": "./chroma_db"
        }
    }
    """
    try:
        if knowledge_loader is None:
            return jsonify({
                "success": False,
                "error": "知识库未初始化"
            }), 500

        stats = knowledge_loader.get_stats()

        logger.info(f"获取知识库统计: {stats}")

        return jsonify({
            "success": True,
            "stats": stats
        })

    except Exception as e:
        logger.error(f"获取统计失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"获取统计信息失败: {str(e)}"
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """
    获取对话历史

    GET /api/history

    Returns:
    {
        "success": true,
        "history": [
            {"role": "user", "content": "用户消息"},
            {"role": "assistant", "content": "AI回复"}
        ],
        "count": 2
    }
    """
    try:
        if agent is None:
            return jsonify({
                "success": False,
                "error": "Agent未初始化"
            }), 500

        history = agent.get_conversation_history()

        return jsonify({
            "success": True,
            "history": history,
            "count": len(history)
        })

    except Exception as e:
        logger.error(f"获取历史失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_memory():
    """
    清空对话记忆

    POST /api/clear

    Returns:
    {
        "success": true,
        "message": "对话记忆已清空"
    }
    """
    try:
        if agent is None:
            return jsonify({
                "success": False,
                "error": "Agent未初始化"
            }), 500

        agent.clear_memory()
        logger.info("对话记忆已清空")

        return jsonify({
            "success": True,
            "message": "对话记忆已清空"
        })

    except Exception as e:
        logger.error(f"清空失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """
    获取系统状态

    GET /api/status

    Returns:
    {
        "success": true,
        "status": {
            "agent_initialized": true,
            "knowledge_base_initialized": true,
            "memory_count": 0,
            "max_memory": 50,
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "api_key_configured": true,
            "knowledge_total": 150,
            "embedding_model": "text2vec-base-chinese",
            "vectorstore": "Chroma",
            "version": "4.0"
        }
    }
    """
    try:
        kb_stats = knowledge_loader.get_stats() if knowledge_loader else {}

        return jsonify({
            "success": True,
            "status": {
                "agent_initialized": agent is not None,
                "knowledge_base_initialized": knowledge_loader is not None,
                "memory_count": agent.get_memory_length() if agent else 0,
                "max_memory": Config.MAX_MEMORY_MESSAGES,
                # "model": Config.OPENAI_MODEL,
                "model": Config.DEEPSEEK_MODEL,
                "temperature": Config.TEMPERATURE,
                # "api_key_configured": bool(Config.OPENAI_API_KEY),
                "api_key_configured": bool(Config.DEEPSEEK_API_KEY),
                "knowledge_total": kb_stats.get('total_documents', 0),
                "embedding_model": "text2vec-base-chinese",
                "vectorstore": "Chroma",
                "version": "4.0"
            }
        })

    except Exception as e:
        logger.error(f"获取状态失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口

    GET /health

    Returns:
    {
        "status": "healthy",
        "service": "Restaurant AI Agent V4",
        "version": "4.0",
        "features": [...]
    }
    """
    return jsonify({
        "status": "healthy",
        "service": "Restaurant AI Agent V4",
        "version": "4.0",
        "features": [
            "Chroma Vector Database",
            "text2vec-base-chinese Embedding",
            "Real Knowledge Base",
            "Semantic Search",
            "RAG Enhanced Responses"
        ]
    })


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        "success": False,
        "error": "API接口不存在"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"服务器内部错误: {error}", exc_info=True)
    return jsonify({
        "success": False,
        "error": "服务器内部错误"
    }), 500


def print_banner():
    """打印启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🍽️  餐饮服务AI Agent V4 - RAG系统                    ║
    ║                                                              ║
    ║         基于真实知识库的智能对话系统                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    📍 服务地址: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    print(f"    📡 API文档: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/health")
    print()
    print("    🔧 技术栈:")
    print("       • 向量数据库: Chroma")
    print("       • Embedding: text2vec-base-chinese")
    print("       • LLM: OpenAI GPT")
    print("       • 知识源: 真实访谈数据")
    print()
    print("    📋 可用端点:")
    print("       POST /api/chat            - 聊天对话")
    print("       POST /api/search_knowledge - 搜索知识库")
    print("       GET  /api/knowledge_stats - 知识库统计")
    print("       GET  /api/history         - 对话历史")
    print("       POST /api/clear           - 清空记忆")
    print("       GET  /api/status          - 系统状态")
    print("       GET  /health              - 健康检查")
    print()
    print("    💡 提示: 按 Ctrl+C 停止服务")
    print("    " + "=" * 62)
    print()


def main():
    """主函数"""
    # 检查配置
    # if not Config.OPENAI_API_KEY:
    #     print("\n❌ 错误：未找到OpenAI API Key")
    #     print("请在项目根目录创建 .env 文件，并添加:")
    #     print("OPENAI_API_KEY=your_api_key_here\n")
    #     return

    if not Config.DEEPSEEK_API_KEY:
        print("\n❌ 错误：未找到DEEPSEEK API Key")
        print("请在项目根目录创建 .env 文件，并添加:")
        print("DEEPSEEK_API_KEY=your_api_key_here\n")
        return

    # 打印启动横幅
    print_banner()

    try:
        # 初始化系统
        initialize_system()

        print("✅ 系统就绪，开始监听请求...\n")

        # 启动Flask应用
        app.run(
            host=Config.FLASK_HOST,
            port=Config.FLASK_PORT,
            debug=Config.FLASK_DEBUG
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        logger.error(f"服务启动失败: {e}", exc_info=True)
        print(f"\n❌ 服务启动失败: {e}")
        print("请检查:")
        print("  1. 知识库文件是否存在: ./data/知识库整理.docx")
        print("  2. 依赖包是否完整安装")
        print("  3. OpenAI API Key是否有效")
        print("  4. 查看上方完整错误日志\n")


if __name__ == "__main__":
    main()
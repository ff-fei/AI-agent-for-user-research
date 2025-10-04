import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    # DeepSeek API配置（替换OpenAI）
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL = "deepseek-reasoner"  # 可用模型：deepseek-chat/deepseek-vl等


    # OpenAI API配置
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # OPENAI_MODEL = "gpt-4o-mini"


    # 对话记忆配置
    MAX_MEMORY_MESSAGES = 50

    # 温度参数（控制回答的随机性，0-1之间）
    TEMPERATURE = 0.7

    # Flask配置（第二版使用）
    FLASK_HOST = "127.0.0.1"
    FLASK_PORT = 5000
    FLASK_DEBUG = True

# 验证API Key
if not Config.DEEPSEEK_API_KEY:
    print("警告：未检测到DEEPSEEK_API_KEY环境变量。请在项目根目录创建.env文件并添加：")
    print("DEEPSEEK_API_KEY=your_deepseek_api_key_here")

# 验证API Key
# if not Config.OPENAI_API_KEY:
#     print("警告：未检测到OPENAI_API_KEY环境变量。请在项目根目录创建.env文件并添加：")
#     print("OPENAI_API_KEY=your_api_key_here")
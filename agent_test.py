from openai import OpenAI, AsyncOpenAI
from agents import OpenAIChatCompletionsModel, Agent, Runner, set_default_openai_client, function_tool
from agents.model_settings import ModelSettings
import os
from dotenv import load_dotenv
from IPython.display import display, Code, Markdown, Image
import requests, json
import os.path

load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

external_client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

set_default_openai_client(external_client)

deepseek_model = OpenAIChatCompletionsModel(
    model=MODEL,
    openai_client=external_client
)

# 创建文件夹的工具函数
@function_tool
def create_folder(folder_name):
    """
    创建一个文件夹
    :param folder_name: 文件夹名称
    :return: 文件夹创建的结果，成功返回提示消息，否则返回错误信息
    """
    try:
        # 检查文件夹是否已存在
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return f"文件夹 '{folder_name}' 创建成功!"
        else:
            return f"文件夹 '{folder_name}' 已经存在!"
    except Exception as e:
        return f"发生错误：{e}"

# 创建文件的工具函数
@function_tool 
def create_file(file_path, content=""):     
    """     
    创建一个文件并写入内容     
    :param file_path: 文件路径，可以是相对路径或绝对路径
    :param content: 文件内容，默认为空字符串     
    :return: 文件创建的结果，成功返回提示消息，否则返回错误信息     
    """     
    try:
        # 确保文件所在目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # 检查文件是否已存在         
        if not os.path.exists(file_path):            
            with open(file_path, "w") as file:                 
                file.write(content)             
            return f"文件 '{file_path}' 创建成功!"     
        else:             
            return f"文件 '{file_path}' 已经存在!"     
    except Exception as e:         
        return f"发生错误：{e}"

# 创建一个复合操作助手
complex_agent = Agent(
    name="ComplexAssistant", 
    instructions="""你是一个能够处理复合文件操作请求的助手。你可以：
    1. 创建文件夹
    2. 创建文件并写入内容
    3. 在指定文件夹中创建文件
    
    对于用户的复合请求，请按照正确的顺序执行操作。例如，如果用户要求在新文件夹中创建文件，
    你应该先创建文件夹，然后再在该文件夹中创建文件。
    
    分析用户请求，提取出所有需要执行的操作，然后按正确的顺序调用相应的工具函数。
    
    请确保准确理解用户的路径需求，例如"在1文件夹下创建test.txt"意味着路径应该是"1/test.txt"。
    """,
    tools=[create_folder, create_file],
    model=deepseek_model
)

# 异步函数
async def chat(Agent):
    input_items = []
    while True:
        try:
            user_input = input("💬 请输入你的消息（输入quit退出）：")
            if user_input.lower() in ["exit", "quit"]:
                print("✅ 对话已结束")
                return
            
            input_items.append({"content": user_input, "role": "user"})
            
            # 运行智能体并处理工具调用
            result = await Runner.run(Agent, input_items)
            
            display(Markdown(result.final_output))
            input_items = result.to_input_list()
            
        except Exception as e:
            print(f"发生错误：{e}")
            return

# 异步函数
async def main():
    try:
        await chat(complex_agent)
    finally:
        # 关闭客户端连接
        try:
            print("正在关闭连接...")
            await external_client.close()
            print("连接已关闭")
        except Exception as e:
            print(f"关闭客户端时发生错误: {e}")

if __name__ == '__main__':
    # 运行异步函数
    import asyncio
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"程序运行时发生错误：{e}")
    finally:
        # 确保程序退出
        print("程序结束")
        os._exit(0)
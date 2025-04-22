import asyncio
import os
import json
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack

from openai import OpenAI  
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 加载 .env 文件，确保 API Key 受到保护
load_dotenv()

class MCPMoleculeClient:
    def __init__(self):
        """初始化分子生成和对接 MCP 客户端"""
        self.exit_stack = AsyncExitStack()
        self.api_key = os.getenv("API_KEY")  # 读取 OpenAI API Key
        self.base_url = os.getenv("BASE_URL")  # 读取 BASE URL，默认为OpenAI
        self.model = os.getenv("MODEL")  # 读取 model，默认为gpt-4

        # print(f"api_key = {self.api_key}, base_url = {self.base_url}, model = {self.model}")
        
        if not self.api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 API_KEY")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)  # 创建OpenAI client
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()        

    async def connect_to_server(self, server_script_path: str):
        """连接到分子生成和对接 MCP 服务器并列出可用工具"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # 启动 MCP 服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # 列出 MCP 服务器上的工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器，支持以下工具:", [tool.name for tool in tools])
        
        # 打印每个工具的详细信息，包括输入参数
        # for tool in tools:
        #     print(f"\n工具名称: {tool.name}")
        #     print(f"描述: {tool.description}")
        #     print("输入参数:")
        #     if tool.inputSchema:
        #         schema = tool.inputSchema
        #         if 'properties' in schema:
        #             for param_name, param_info in schema['properties'].items():
        #                 param_type = param_info.get('type', '未指定')
        #                 param_desc = param_info.get('description', '无描述')
        #                 print(f"  - {param_name} ({param_type}): {param_desc}")
        
    async def process_query(self, query: str) -> str:
        """
        使用大模型处理查询并调用分子生成或对接相关的 MCP 工具
        """
        # 检查输入是否可能是直接的文件路径
        pdb_file = None
        sdf_file = None
        pdbqt_file = None
        
        # 简单检测可能的文件路径
        if '.pdb' in query:
            words = query.split()
            for word in words:
                if word.endswith('.pdb') and os.path.exists(word):
                    pdb_file = word
                    print(f"检测到PDB文件路径: {pdb_file}")
        
        if '.sdf' in query:
            words = query.split()
            for word in words:
                if word.endswith('.sdf') and os.path.exists(word):
                    sdf_file = word
                    print(f"检测到SDF文件路径: {sdf_file}")
        
        if '.pdbqt' in query:
            words = query.split()
            for word in words:
                if word.endswith('.pdbqt') and os.path.exists(word):
                    pdbqt_file = word
                    print(f"检测到PDBQT文件路径: {pdbqt_file}")
        
        # 设置系统提示，引导模型更好地理解分子生成和对接相关查询
        system_prompt = """
        你是一个专业的分子生成和对接助手，可以帮助用户处理分子对接和生成任务。
        你可以调用以下工具:

        1. molecule_generation - 用于执行分子生成计算，需要提供PDB文件路径，可选参数包括:
            - ref_ligand: 可以是"A:330"(默认值，无参考配体模式)或SDF文件绝对路径(有参考配体模式)
            - n_samples: 生成样本数量，默认为2
        2. download_molecule - 用于下载生成的分子文件，需要提供分子文件名，可选提供保存路径
        3. molecular_docking - 用于执行分子对接计算，需要提供配体SDF文件路径、受体PDB文件路径和对接模式
        4. download_docking_result - 用于下载对接结果文件，需要提供结果文件名，可选提供保存路径
        5. conformation_evaluation - 用于执行构象评估计算，需要提供预测构象文件路径、真实构象文件路径、条件蛋白质文件路径和对接模式
        6. download_evaluation_result - 用于下载构象评估结果文件，需要提供结果文件名，可选提供保存路径
        7. batch_download_docking_results - 用于批量下载对接结果文件，需要提供结果文件名列表和保存目录路径

        
        当用户提供PDB文件路径时，请调用molecule_generation工具并将参数格式设置为{'params': {'pdb_file': '用户提供的路径'}}。
        如果用户想要无参考配体的分子生成，可以使用默认的ref_ligand="A:330"或不指定此参数。
        如果用户想要有参考配体的分子生成，ref_ligand应该是一个.sdf文件的路径。
        
        当用户请求下载分子时，请调用download_molecule工具并将参数格式设置为{'params': {'molecule_name': '分子文件名', 'output_path': '保存路径'}}。
        分子文件名通常是PDB ID加上_mol.sdf的形式，例如"3rfm_mol.sdf"。
        
        当用户请求分子对接时，请调用molecular_docking工具并将参数格式设置为{'params': {'ligand_sdf': '配体文件路径', 'protein_pdb': '受体文件路径', 'dock_mode': '对接模式'}}。
        对接模式可以是"adgpu"或"vina"。
        
        当用户请求下载对接结果时，请调用download_docking_result工具并将参数格式设置为{'params': {'result_file': '结果文件名', 'output_path': '保存路径'}}。
        对于adgpu模式，结果文件名通常是PDB ID加上_ligand_X_Y.pdbqt的形式，例如"3rfm_ligand_0_1.pdbqt"。
        对于vina模式，结果文件名通常是PDB ID加上_ligand_Z.pdbqt的形式，例如"3rfm_ligand_2.pdbqt"。

        当用户请求批量下载对接结果时，请调用batch_download_docking_results工具并将参数格式设置为{'params': {'result_files': ['文件名1', '文件名2', ...], 'output_dir': '保存目录路径'}}。
        这个工具可以一次下载分子对接生成的所有结果文件。
        
        当用户请求构象评估时，请调用conformation_evaluation工具并将参数格式设置为{'params': {'pred_file': '预测构象文件路径', 'true_file': '真实构象文件路径', 'cond_file': '条件蛋白质文件路径', 'dock_mode': '对接模式'}}。
        预测和真实构象文件通常是.pdbqt格式，条件蛋白质文件通常是.pdb格式。
        
        当用户请求下载评估结果时，请调用download_evaluation_result工具并将参数格式设置为{'params': {'result_file': '结果文件名', 'output_path': '保存路径'}}。
        评估结果文件通常是"posebusters_results.csv"。

        
        
        请理解用户的意图，并调用适当的工具来完成任务。确保参数格式正确。
        
        典型的工作流程有三种：
        
        分子生成流程：
        1. 用户提供PDB文件
        2. 调用molecule_generation生成分子
        3. 调用download_molecule下载生成的分子文件
        
        分子对接流程：
        1. 用户提供配体SDF文件和受体PDB文件
        2. 调用molecular_docking执行分子对接
        3. 调用download_docking_result下载对接结果文件
        
        构象评估流程：
        1. 用户提供预测构象PDBQT文件、真实构象PDBQT文件和条件蛋白质PDB文件
        2. 调用conformation_evaluation执行构象评估
        3. 调用download_evaluation_result下载评估结果文件
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = await self.session.list_tools()
            
            # 打印所有可用工具及其参数结构以便调试
            # print("可用工具列表:")
            # for tool in response.tools:
            #     print(f"- {tool.name}: {tool.inputSchema}")
            
            # 手动构造 tools 参数，符合 API 的要求
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "molecule_generation",
                        "description": "执行分子生成计算，需要提供 PDB 文件路径",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "pdb_file": {
                                            "type": "string",
                                            "description": "受体文件的绝对路径，必须为 .pdb 格式"
                                        },
                                        "ref_ligand": {
                                            "type": "string",
                                            "description": "参考配体信息，可以是\"A:330\"（默认值，无参考配体）或者SDF文件的绝对路径"
                                        },
                                        "n_samples": {
                                            "type": "integer",
                                            "description": "生成样本数量，默认为2"
                                        }
                                    },
                                    "required": ["pdb_file"]
                                }
                            },
                            "required": ["params"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "download_molecule",
                        "description": "下载生成的分子文件",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "molecule_name": {
                                            "type": "string",
                                            "description": "要下载的分子文件名（如 '3rfm_mol.sdf'）"
                                        },
                                        "output_path": {
                                            "type": "string",
                                            "description": "保存文件的本地路径（可选）"
                                        }
                                    },
                                    "required": ["molecule_name"]
                                }
                            },
                            "required": ["params"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "molecular_docking",
                        "description": "执行分子对接计算，需要提供配体SDF文件、受体PDB文件和对接模式",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "ligand_sdf": {
                                            "type": "string",
                                            "description": "配体文件的绝对路径，必须为 .sdf 格式"
                                        },
                                        "protein_pdb": {
                                            "type": "string",
                                            "description": "受体文件的绝对路径，必须为 .pdb 格式"
                                        },
                                        "dock_mode": {
                                            "type": "string",
                                            "description": "对接模式，可选值为'adgpu'或'vina'",
                                            "enum": ["adgpu", "vina"]
                                        }
                                    },
                                    "required": ["ligand_sdf", "protein_pdb", "dock_mode"]
                                }
                            },
                            "required": ["params"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "download_docking_result",
                        "description": "下载分子对接结果文件",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "result_file": {
                                            "type": "string",
                                            "description": "要下载的对接结果文件名"
                                        },
                                        "output_path": {
                                            "type": "string",
                                            "description": "保存文件的本地路径（可选）"
                                        }
                                    },
                                    "required": ["result_file"]
                                }
                            },
                            "required": ["params"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "batch_download_docking_results",
                        "description": "批量下载分子对接结果文件",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "result_files": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "要下载的对接结果文件名列表"
                                        },
                                        "output_dir": {
                                            "type": "string",
                                            "description": "保存文件的目录路径"
                                        }
                                    },
                                    "required": ["result_files", "output_dir"]
                                }
                            },
                            "required": ["params"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "conformation_evaluation",
                        "description": "执行构象评估计算",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "pred_file": {
                                            "type": "string",
                                            "description": "预测的构象文件路径（pdbqt格式）"
                                        },
                                        "true_file": {
                                            "type": "string",
                                            "description": "参考的真实构象文件路径（pdbqt格式）"
                                        },
                                        "cond_file": {
                                            "type": "string",
                                            "description": "条件蛋白质文件路径（pdb格式）"
                                        },
                                        "dock_mode": {
                                            "type": "string",
                                            "description": "对接模式，可选值为'adgpu'或'vina'",
                                            "enum": ["adgpu", "vina"]
                                        }
                                    },
                                    "required": ["pred_file", "true_file", "cond_file", "dock_mode"]
                                }
                            },
                            "required": ["params"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "download_evaluation_result",
                        "description": "下载构象评估结果文件",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "result_file": {
                                            "type": "string",
                                            "description": "要下载的评估结果文件名（如 'posebusters_results.csv'）"
                                        },
                                        "output_path": {
                                            "type": "string",
                                            "description": "保存文件的本地路径（可选）"
                                        }
                                    },
                                    "required": ["result_file"]
                                }
                            },
                            "required": ["params"]
                        }
                    }
                }
            ]
            
            # 打印发送的请求参数以便调试
            # print("发送的请求参数：")
            # print(json.dumps({
            #     "model": self.model,
            #     "messages": messages,
            #     "tools": available_tools,
            #     "tool_choice": "auto"
            # }, indent=2, ensure_ascii=False))
            
            # 调用 OpenAI API 进行工具调用
            # print(f"发送请求给大模型: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,            
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )
            
            # 处理返回的内容
            content = response.choices[0]
            print(f"模型响应类型: {content.finish_reason}")
            
            if content.message.tool_calls:
                # 如果需要使用工具，解析工具调用
                tool_call = content.message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                print(f"\n\n[正在调用工具 {tool_name}，参数: {tool_args}]\n")
                
                # 执行工具
                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                    print(f"工具调用结果: {result}")
                    
                    # 将模型返回的调用工具数据和工具执行结果都存入messages中
                    messages.append(content.message.model_dump())
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result.content[0].text) if isinstance(result.content[0].text, dict) else result.content[0].text,
                        "tool_call_id": tool_call.id,
                    })
                    
                    # 将工具调用结果再返回给大模型用于生成最终结果
                    final_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    )
                    return final_response.choices[0].message.content
                    
                except Exception as e:
                    error_msg = f"工具调用失败: {str(e)}"
                    print(f"\n⚠️ {error_msg}")
                    return error_msg
            else:
                print("模型没有调用工具，直接返回响应")
                
            return content.message.content
        except Exception as e:
            error_msg = f"处理查询时发生错误: {str(e)}"
            print(f"\n⚠️ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
    
    async def chat_loop(self):
        """运行交互式分子生成和对接聊天循环"""
        print("\n🧪 分子生成与对接 MCP 客户端已启动！输入 'quit' 退出")
        print("💡 提示: 你可以请求执行分子生成，例如 '请使用位于/home/zhangfn/workflow/3rfm.pdb的蛋白质文件生成分子'或者'请使用位于/home/zhangfn/workflow/3rfm.pdb的蛋白质文件和位于/home/zhangfn/workflow/3rfm_ligand_0_1_processed.sdf的参考配体生成分子'")
        print("💡 提示: 生成后可请求下载分子，例如 '请下载3rfm_mol.sdf分子到/home/zhangfn/workflow/3rfm_mol.sdf'")
        print("💡 提示: 你可以请求执行分子对接，例如 '请使用/home/zhangfn/workflow/3rfm_mol.sdf作为配体和/home/zhangfn/workflow/3rfm.pdb作为受体进行adgpu对接'")
        print("💡 提示: 对接后可请求下载结果，例如 '请下载3rfm_ligand_0_1.pdbqt对接结果文件，如请下载3rfm_ligand_0_1.pdbqt对接结果到/home/zhangfn/workflow/3rfm_ligand_0_1.pdbqt'")
        print("💡 提示: 对接后也可以请求批量下载结果，例如 '请批量下载所有对接结果文件（文件名以3rfm_ligand_0_*_1.pdbqt格式的，从3rfm_ligand_0_1.pdbqt到3rfm_ligand_1_1.pdbqt、3rfm_ligand_2_1.pdbqt、3rfm_ligand_3_1.pdbqt一直到3rfm_ligand_19_1.pdbqt的20个文件）到/home/zhangfn/workflow目录'")
        print("💡 提示: 你可以请求执行构象评估，例如 '请使用/home/zhangfn/workflow/3rfm_ligand_0_1.pdbqt作为预测构象，/home/zhangfn/workflow/3rfm_ligand_0.pdbqt作为真实构象，/home/zhangfn/workflow/3rfm.pdb作为条件蛋白质进行adgpu模式的构象评估'")
        print("💡 提示: 评估后可请求下载结果，例如 '请下载posebusters_results.csv评估结果到/home/zhangfn/workflow/posebusters_results.csv'")
        

        while True:
            try:
                query = input("\n你: ").strip()
                if query.lower() in ['quit', 'exit', '退出']:
                    break
                
                response = await self.process_query(query)
                print(f"\n🤖 助手: {response}")

            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("使用方法: python molecule_client.py <服务器脚本路径>")
        sys.exit(1)

    client = MCPMoleculeClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
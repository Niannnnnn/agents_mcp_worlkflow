from openai import OpenAI, AsyncOpenAI
from agents import OpenAIChatCompletionsModel, Agent, Runner, set_default_openai_client, function_tool
from agents.model_settings import ModelSettings
from agents.mcp import MCPServer, MCPServerStdio
import os
from dotenv import load_dotenv
from IPython.display import display, Code, Markdown, Image
import requests, json, re
import os.path
from typing import Dict, Any, List, Optional
import asyncio

load_dotenv(override=True)
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
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

class TaskPlanner:
    def __init__(self, model):
        # 创建一个专用于任务规划的Agent
        self.planner_agent = Agent(
            name="TaskPlannerAgent",
            instructions="""你是一个任务规划专家，负责将用户的复杂请求分解为有序的任务步骤。""",
            model=model
        )
        
    async def create_plan(self, user_query: str, feedback: Optional[str] = None) -> list:
        planning_prompt = f"""
        请分析以下用户请求，并将其分解为明确的按顺序执行的任务步骤：
        
        用户请求: {user_query}
        
        请识别需要执行的操作，并按照以下格式返回执行计划:
        {{
          "tasks": [
            {{
              "task_id": 1,
              "operation": "molecule_generation",
              "description": "执行分子生成",
              "parameters": {{
                "param1": "值1"  // 如果用户提供了参数则填写，没有则设为空对象 {{}}
              }}
            }}
          ]
        }}
        
        规则：
        1. 如果用户请求包含多个步骤，必须按照以下顺序排列：先分子生成，再分子对接，然后再构象评估，最后进行结果文件的下载
        2. 不要添加用户没有明确要求的操作
        3. 确保每个操作都有正确的操作类型名称
        4. 只包含用户明确提供的参数，不要臆测参数值
        5. 如果用户要求生成多个分子，合并为一个分子生成任务，通过n_samples参数指定数量
        6. 重要: 不要在JSON中使用注释，如果参数为空则使用 {{}} 空对象
        """
        
        # 如果有反馈，添加到规划提示中
        if feedback:
            planning_prompt += f"""
            以下是上一轮执行的反馈，请据此调整新的执行计划：
            {feedback}
            """

        planning_input = [{"content": planning_prompt, "role": "user"}]
        plan_result = await Runner.run(self.planner_agent, planning_input)
        
        try:
            # 提取JSON部分
            plan_text = plan_result.final_output
            
            # 如果文本包含代码块，提取代码块内容
            if "```json" in plan_text:
                plan_text = plan_text.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_text:
                plan_text = plan_text.split("```")[1].split("```")[0].strip()
            
            # 预处理：移除JSON中的注释行
            cleaned_lines = []
            for line in plan_text.split('\n'):
                if '//' not in line:  # 过滤掉包含注释的行
                    cleaned_lines.append(line)
            
            cleaned_json = '\n'.join(cleaned_lines)
            
            # 解析JSON
            plan = json.loads(cleaned_json)
            return plan["tasks"]
        except Exception as e:
            print(f"\033[91m解析计划时出错: {e}\033[0m")
            print(f"原始计划响应: {plan_result.final_output}")
            
            # 失败后尝试更强力的方式提取任务
            try:
                # 手动提取关键任务信息
                tasks = []
                if "molecule_generation" in plan_result.final_output:
                    tasks.append({
                        "task_id": 1,
                        "operation": "molecule_generation",
                        "description": "执行分子生成",
                        "parameters": {"input_pdb": "/home/zhangfn/workflow/3rfm.pdb", "n_samples": 2}
                    })
                
                if "molecular_docking" in plan_result.final_output and "vina" in plan_result.final_output:
                    tasks.append({
                        "task_id": 2,
                        "operation": "molecular_docking",
                        "description": "执行vina模式的分子对接",
                        "parameters": {}
                    })
                
                if "conformation_evaluation" in plan_result.final_output:
                    tasks.append({
                        "task_id": 3,
                        "operation": "conformation_evaluation",
                        "description": "进行构象评估",
                        "parameters": {}
                    })
                
                if "download_all_outputs" in plan_result.final_output:
                    tasks.append({
                        "task_id": 4,
                        "operation": "download_all_outputs",
                        "description": "下载结果文件",
                        "parameters": {}
                    })

                if tasks:
                    print(f"\033[93m通过备用方式生成了任务计划\033[0m")
                    return tasks
            except:
                pass
                
            # 返回一个空计划
            return []



async def run_agent_until_done(executor_agent, input_items, tasks=None):
    """按照规划执行任务，直到所有任务完成"""
    results = []
    
    if tasks:
        # 如果提供了任务列表，则按照计划执行
        for task in tasks:
            task_desc = task["description"]
            operation = task["operation"]
            parameters = task.get("parameters", {})
            
            # 构建任务描述
            task_prompt = f"执行 {operation} 操作"
            if parameters:
                param_str = ", ".join([f"{k}={v}" for k, v in parameters.items() if v])
                task_prompt += f"，参数: {param_str}"
            else:
                task_prompt += "，使用默认参数"
            
            print(f"\033[94m正在执行任务: {task_desc}\033[0m")
            
            # 构建更明确的指令，防止重复调用
            task_input = [{"content": f"""
            请执行以下单个任务，并且只调用一次相关工具：
            
            {task_prompt}
            
            注意：只需调用一次工具函数，完成后立即返回结果，不要重复调用。
            """, "role": "user"}]
            
            # 执行任务
            result = await Runner.run(executor_agent, task_input)
            
            # 保存结果
            results.append({
                "task_id": task["task_id"],
                "operation": operation,
                "description": task_desc,
                "result": result.final_output
            })
            
            print(f"\033[92m✓ 完成任务: {task_desc}\033[0m")
    else:
        # 如果没有提供任务列表，则直接执行输入
        result = await Runner.run(executor_agent, input_items)
        results.append({
            "task_id": 1,
            "operation": "direct_execution",
            "description": "执行用户请求",
            "result": result.final_output
        })
    
    # 返回所有结果的组合
    return results

def parse_feedback_items(feedback):
    for idx, item in enumerate(feedback.new_items):
        

        # 专盯 New Item 1：包含 function_call_output 且含有 json 的 text
        if idx == 1 and isinstance(item.raw_item, dict):
            output_str = item.raw_item.get("output", "")
            try:
                # 提取 JSON 字符串里的 text
                json_data = json.loads(output_str)
                text_str = json_data.get("text", "")
                
                # 把嵌套的 JSON 字符串转成 dict
                inner_data = json.loads(text_str)
                
                # # 渲染输出
                # print(f"--- New Item {idx} ---")
                # print(f"🧪 评估状态: {inner_data['status']}")
                # print(f"📝 总结信息: {inner_data['message']}")

                # 将结果存入字典
                result = {
                    "status": inner_data.get("status", "未找到状态"),
                    "message": inner_data.get("message", "未找到信息")
                }
                return result  # 直接返回结果并退出函数

            except Exception as e:
                print("⚠️ 解析出错啦：", e)
    
    # 如果没有找到匹配的item
    result["error"] = "未找到索引为1的有效项目"
    return result

async def chat(mcp_servers: list[MCPServer]):
    # 创建规划智能体
    planner_agent = TaskPlanner(deepseek_model)

    # 创建一个执行智能体
    executor_agent = Agent(
        name="ExecutorAgent", 
        instructions="""你是一个能够执行分子生成、分子对接、构象评估操作的分子设计工作流的助手。你可以：
        1. 执行分子生成操作
        2. 执行分子对接计算
        3. 构象评估
        
        分析用户请求，根据当前正在执行的单一操作任务，执行对应的工具函数。
        每个任务只能调用一次相应的工具函数，避免重复调用。
        
        重要提示：

        1. 如果用户需要分子生成，使用molecule_generation这个工具。如果用户说"生成 2 个分子"，应该设置 n_samples = 2。

        2. 如果用户需要执行分子对接计算，使用molecular_docking工具。如果用户没有提供配体和受体的路径，使用默认路径。

        3. 如果用户需要进行构象评估，则使用conformation_evaluation工具。如果用户没有提供pred_file, cond_file参数的路径，使用默认路径。

        4. 每次只专注于当前被指派的单个任务。你现在只需要执行系统给你的当前一个任务，不需要考虑全局任务流程。
        
        5. 请不要自作主张地添加参数，如果用户没有明确提供参数值，就使用工具的默认值。
        
        6. 如果用户需要进行文件下载，则使用download_all_outputs工具。如果用户没有输出路径，使用默认路径。
        
        7. 对于每个任务，严格只调用一次对应的工具，完成后立即返回结果。
        """,
        mcp_servers=mcp_servers,
        model=deepseek_model
    )
    # 创建一个反馈智能体
    reflection_agent = Agent(
        name="ExecutorAgent", 
        instructions="""你是一个分子设计工作流的反馈分析专家，负责分析任务执行结果，提供优化建议和错误诊断。
        你只需要使用molecule_reflection工具获取评估结果，并基于评估结果:
        1. 对后续任务规划提供优化建议
        
        重要提示：在众多mcp_servers对应的工具中，你只需要使用molecule_reflection工具，你不会用到其他工具，请不要错误调用。

        你的反馈将用于指导下一轮任务规划，帮助用户获得更好的分子设计结果。
        """,
        mcp_servers=mcp_servers,
        model=deepseek_model
    )

    input_items = []

    
    # 打印欢迎信息和使用提示
    print("\n====== 分子设计工作流助手 ======")
    print("本助手可以帮您完成以下任务：")
    print("1. 分子生成 - 根据受体结构生成小分子")
    print("2. 分子对接 - 使用adgpu或vina模式进行分子对接")
    print("3. 构象评估 - 进行构象合理性评估")
    print("4. 文件下载 - 下载执行操作的结果文件")
    print("5. 结果反馈 - 分析执行结果并提供优化建议")
    print("\n您可以通过自然语言描述需要执行的任务，下面是一些示例：")
    
    # 定义示例提示，并用颜色高亮显示
    examples = [
        "请使用/home/zhangfn/workflow/3rfm.pdb生成2个分子",
        "请执行vina模式的分子对接，使用/home/zhangfn/workflow/3rfm_mol.sdf作为配体，/home/zhangfn/workflow/3rfm.pdb作为受体",
        "请使用/home/zhangfn/test_file/3rfm_ligand_0_vina.pdbqt作为pred_file，/home/zhangfn/workflow/3rfm.pdb作为cond_file，vina作为dock_mode进行构象评估",
        "请先使用/home/zhangfn/workflow/3rfm.pdb生成4个分子，再进行分子对接，然后再进行构象评估",
        "请先使用/home/zhangfn/workflow/3rfm.pdb生成2个分子，再进行vina模式的分子对接，然后再进行构象评估。最后将结果文件下载到/home/zhangfn/test_download"
    ]
    
    # 打印带颜色的示例
    for i, example in enumerate(examples):
        print(f"\033[96m示例{i+1}: {example}\033[0m")
    
    print("\n" + "="*35)
    
    # 保存上一次反馈，用于改进后续规划
    last_feedback = None

    while True:
        try:
            if last_feedback is not None:
                print(f"last_feedback : {last_feedback}")
            print("\n您可以输入需要执行的任务，或输入'help'查看帮助信息：")
            user_input = input("\033[95m💬 请输入您的指令：\033[0m ")
            
            # 处理特殊命令
            if user_input.lower() in ["exit", "quit"]:
                print("\033[92m✅ 对话已结束\033[0m")
                os._exit(0)
            elif user_input.lower() == "help":
                print("\n\033[93m==== 帮助信息 ====\033[0m")
                print("您可以执行以下操作：")
                print("- 分子生成: 指定受体PDB文件路径和可选的参考配体")
                print("- 分子对接: 指定配体SDF文件、受体PDB文件和对接模式(adgpu/vina)")
                print("- 构象评估: 指定预测SDF文件、蛋白PDB文件和对接模式(adgpu/vina)")
                print("- 完整工作流: 一步执行从分子生成到对接结果下载的全流程")
                print("- 结果反馈: 分析执行结果并提供优化建议")
                print("\n示例命令：")
                for i, example in enumerate(examples):
                    print(f"\033[96m示例{i+1}: {example}\033[0m")
                continue
            
            # 保存用户输入
            input_items = [{"content": user_input, "role": "user"}]
            
            # 显示处理中提示
            print("\033[93m正在规划任务执行流程...\033[0m")
            
            # 创建执行计划
            tasks = await planner_agent.create_plan(user_input, last_feedback)
            
            if tasks:
                # 打印计划
                print("\033[94m执行计划:\033[0m")
                for idx, task in enumerate(tasks):
                    print(f"\033[94m{idx+1}. {task['description']}\033[0m")
                
                # 按照计划执行任务
                results = await run_agent_until_done(executor_agent, input_items, tasks)
                
                # 显示所有任务结果
                print(f"\033[92m✅ 全部任务执行完成!\033[0m")
                print(f"\n执行结果概要:")
                
                for result in results:
                    print(f"\n\033[94m[任务 {result['task_id']}] {result['description']}:\033[0m")
                    print(f"{result['result']}")

                # 使用反馈智能体分析结果 - 不再传入执行结果，让它直接调用API
                print("\n\033[93m正在分析执行结果...\033[0m")
                feedback_input = [{"role": "user", "content": "feedback"}]
                feedback = await Runner.run(reflection_agent, feedback_input)
                
                # 显示反馈结果
                print(f"\n\033[94m[执行反馈]:\033[0m")
                last_feedback = parse_feedback_items(feedback)  # 保存反馈用于下一次规划

                # print(f"last_feedback : {last_feedback}")
                
            else:
                # 如果无法创建计划，直接执行单次任务
                print("\033[93m无法创建明确的执行计划，将直接处理请求...\033[0m")
                results = await run_agent_until_done(executor_agent, input_items)
                
                print(f"\033[92m✅ 执行完成!\033[0m")
                if results and len(results) > 0:
                    print(f"\n执行结果概要:\n{results[0]['result']}")
            
        except KeyboardInterrupt:
            print("\n\033[92m✅ 操作已中断，对话结束\033[0m")
            return
        except Exception as e:
            print(f"\n\033[91m发生错误：{e}\033[0m")
            import traceback
            traceback.print_exc()
            print("\n\033[93m您可以尝试重新输入或使用不同的表达方式\033[0m")

async def mcp_run():
    async with MCPServerStdio(
        name = "molecular_generation_server",
        cache_tools_list = True,
        params = {"command": "uv", "args": ["run", "mol_generation_server.py"]} 
    ) as mol_gen_server, MCPServerStdio(
        name = "molecular_docking_server",
        cache_tools_list = True,
        params = {"command": "uv", "args": ["run", "mol_docking_server.py"]}
    ) as docking_server, MCPServerStdio(
        name = "molecular_eval_server",
        cache_tools_list = True,
        params = {"command": "uv", "args": ["run", "mol_eval_server.py"]}
    ) as conf_eval_server, MCPServerStdio(
        name = "molecular_download_server",
        cache_tools_list = True,
        params = {"command": "uv", "args": ["run", "mol_download_server.py"]}
    ) as mol_download_server, MCPServerStdio(
        name = "molecular_reflection_server",
        cache_tools_list = True,
        params = {"command": "uv", "args": ["run", "mol_reflection_server.py"]}
    ) as mol_reflection_server:
        await chat([mol_gen_server, docking_server, conf_eval_server, mol_download_server, mol_reflection_server])

if __name__ == '__main__':
    asyncio.run(mcp_run())
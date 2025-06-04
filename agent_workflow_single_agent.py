from openai import OpenAI, AsyncOpenAI
from agents import OpenAIChatCompletionsModel, Agent, Runner, set_default_openai_client, function_tool
from agents.model_settings import ModelSettings
import os
from dotenv import load_dotenv
from IPython.display import display, Code, Markdown, Image
import requests, json
import os.path
from typing import Dict, Any, List
import asyncio
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

# 移除function_tool装饰器，变为普通函数
def molecule_generation(pdb_file, ref_ligand="A:330", n_samples=1):
    """执行分子生成计算
    
    Args:
        pdb_file: 受体文件绝对路径（必须为.pdb格式）
        ref_ligand: 参考配体信息，可以是"A:330"（默认值，无参考配体）或者SDF文件的绝对路径
        n_samples: 生成样本数量（可选，默认为1）
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 计算结果或错误信息}
    """
    # 构建params字典
    params = {
        'pdb_file': pdb_file,
        'ref_ligand': ref_ligand,
        'n_samples': n_samples
    }

    print(f"收到分子生成请求，参数: {params}")
    
    # 参数校验
    if not params.get('pdb_file'):
        return {"status": "error", "message": "未提供PDB文件路径"}
    
    pdb_path = params['pdb_file']
    if not os.path.exists(pdb_path):
        return {"status": "error", "message": f"PDB文件不存在: {pdb_path}"}
    
    if not pdb_path.endswith('.pdb'):
        return {"status": "error", "message": f"文件格式错误，必须是.pdb格式: {pdb_path}"}
    
    # 检查参考配体
    ref_ligand = params.get('ref_ligand', 'A:330')
    if ref_ligand != 'A:330' and (not os.path.exists(ref_ligand) or not ref_ligand.endswith('.sdf')):
        return {"status": "error", "message": f"参考配体文件不存在或格式错误(应为.sdf): {ref_ligand}"}
    
    n_samples = params.get('n_samples', 1)
    
    # 构建API请求负载
    try:
        with open(pdb_path, 'rb') as f:
            pdb_content = f.read()
        
        files = {'pdb_file': (os.path.basename(pdb_path), pdb_content)}
        
        data = {'n_samples': n_samples}
        
        # 如果是SDF文件，也读取到内存中
        if ref_ligand != 'A:330' and os.path.exists(ref_ligand):
            with open(ref_ligand, 'rb') as ref_file:
                ref_content = ref_file.read()
            files['ref_ligand_file'] = (os.path.basename(ref_ligand), ref_content)
        else:
            data['ref_ligand'] = ref_ligand
        
        # 调用Flask API
        print(f"正在调用分子生成API...")
        response = requests.post(
            "http://localhost:5000/api/molecule_generation",
            files=files,
            data=data,
            timeout=300
        )
            
        print(f"API响应: {response.text}")
        if response.status_code == 200:
            result = response.json()
            # 从下载URL中提取分子文件名
            molecule_name = os.path.basename(result.get('download_url', ''))
            result['molecule_name'] = molecule_name  # 添加分子名称到结果中
            return {
                "status": "success", 
                "message": "分子生成计算完成",
                "result": result
            }
        else:
            return {
                "status": "error", 
                "message": f"API返回错误: {response.status_code}", 
                "response": response.text
            }
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return {"status": "error", "message": f"API调用失败: {str(e)}"}

# 移除function_tool装饰器，变为普通函数
def download_molecule(molecule_name, output_path):
    """下载生成的分子文件
    
    Args:
        molecule_name: 分子文件名 (如 "3rfm_mol.sdf")
        output_path: 保存文件的本地路径 (可选，默认为当前目录下的同名文件)
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "message": 操作结果或错误信息, "file_path": 保存的文件路径}
    """
    print(f"收到分子下载请求，参数: molecule_name={molecule_name}, output_path={output_path}")
    
    # 参数校验
    if not molecule_name:
        return {"status": "error", "message": "未提供分子文件名"}
    
    # 获取输出路径，如未提供则使用当前目录和原始文件名
    if not output_path:
        output_path = os.path.join(os.getcwd(), molecule_name)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            return {"status": "error", "message": f"无法创建输出目录: {str(e)}"}
    
    # 构建下载URL
    try:
        download_url = f"http://localhost:5000/api/download/molecule_generation/{molecule_name}"
        
        print(f"正在从 {download_url} 下载分子文件...")
        response = requests.get(download_url, stream=True, timeout=300)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "status": "success",
                "message": f"分子文件成功下载到 {output_path}",
                "file_path": output_path
            }
        else:
            return {
                "status": "error",
                "message": f"下载失败，服务器返回: {response.status_code}",
                "response": response.text
            }
    except Exception as e:
        print(f"分子下载失败: {str(e)}")
        return {"status": "error", "message": f"下载失败: {str(e)}"}

def molecular_docking(ligand_sdf, protein_pdb, dock_mode):
    """执行分子对接计算
    
    Args:
        ligand_sdf: 配体文件绝对路径（必须为.sdf格式）
        protein_pdb: 受体文件绝对路径（必须为.pdb格式）
        dock_mode: 对接模式，可选值为"adgpu"或"vina"
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 计算结果或错误信息, "result_files": 结果文件列表}
    """
    
    # 构建params字典
    params = {
        'ligand_sdf': ligand_sdf,
        'protein_pdb': protein_pdb,
        'dock_mode': dock_mode
    }

    print(f"收到分子对接请求，参数: {params}")
    
    # 参数校验
    if not params.get('ligand_sdf'):
        return {"status": "error", "message": "未提供配体SDF文件路径"}
    
    if not params.get('protein_pdb'):
        return {"status": "error", "message": "未提供受体PDB文件路径"}
    
    if not params.get('dock_mode'):
        return {"status": "error", "message": "未提供对接模式"}
    
    ligand_path = params['ligand_sdf']
    protein_path = params['protein_pdb']
    dock_mode = params['dock_mode']
    
    # 文件存在性检查
    if not os.path.exists(ligand_path):
        return {"status": "error", "message": f"配体文件不存在: {ligand_path}"}
    
    if not os.path.exists(protein_path):
        return {"status": "error", "message": f"受体文件不存在: {protein_path}"}
    
    # 文件格式检查
    if not ligand_path.endswith('.sdf'):
        return {"status": "error", "message": f"配体文件格式错误，必须是.sdf格式: {ligand_path}"}
    
    if not protein_path.endswith('.pdb'):
        return {"status": "error", "message": f"受体文件格式错误，必须是.pdb格式: {protein_path}"}
    
    # 对接模式检查
    if dock_mode not in ['adgpu', 'vina']:
        return {"status": "error", "message": f"对接模式错误，必须是'adgpu'或'vina': {dock_mode}"}
    
    # 构建API请求负载
    try:
        with open(ligand_path, 'rb') as ligand_file, open(protein_path, 'rb') as protein_file:
            files = {
                'ligand_sdf': (os.path.basename(ligand_path), ligand_file.read()),
                'protein_pdb': (os.path.basename(protein_path), protein_file.read())
            }
            
            data = {
                'dock_mode': dock_mode
            }
            
            # 调用Flask API
            print(f"正在调用分子对接API，模式: {dock_mode}...")
            response = requests.post(
                "http://localhost:5000/api/molecular_docking",
                files=files,
                data=data,
                timeout=600  # 根据计算时长调整，分子对接可能需要更长时间
            )
            
            print(f"API响应: {response.text}")
            if response.status_code == 200:
                result = response.json()
                
                # 确保从API响应中正确提取结果文件列表
                result_files = result.get('result_files', [])
                
                # 如果API未返回文件列表，则从结果中提取
                if not result_files and 'download_urls' in result:
                    result_files = [os.path.basename(url) for url in result['download_urls']]
                
                print(f"提取到的结果文件列表: {result_files}")
                
                return {
                    "status": "success", 
                    "message": f"分子对接计算完成 ({dock_mode}模式)",
                    "result": result,
                    "result_files": result_files  # 确保返回文件列表
                }
            else:
                return {
                    "status": "error", 
                    "message": f"API返回错误: {response.status_code}", 
                    "response": response.text
                }
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return {"status": "error", "message": f"API调用失败: {str(e)}"}

def batch_download_docking_results(result_files, output_dir):
    """批量下载分子对接结果文件
    
    Args:
        params: 包含以下字段的字典:
            result_files: 结果文件名列表
            output_dir: 保存文件的目录路径
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "message": 操作结果或错误信息, "downloaded": 成功下载的文件列表, "failed": 下载失败的文件列表}
    """
    # 构建params字典
    params = {
        'result_files': result_files,
        'output_dir': output_dir
    }
    
    print(f"收到批量对接结果下载请求，参数: {params}")
    
    # 参数校验
    result_files = params.get('result_files')
    if not result_files:
        return {"status": "error", "message": "未提供结果文件名列表"}
    
    output_dir = params.get('output_dir')
    if not output_dir:
        return {"status": "error", "message": "未提供输出目录路径"}
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            return {"status": "error", "message": f"无法创建输出目录: {str(e)}"}
    
    # 逐个下载文件
    downloaded_files = []
    failed_files = []
    
    for result_file in result_files:
        try:
            download_url = f"http://localhost:5000/api/download/molecular_docking/{result_file}"
            output_path = os.path.join(output_dir, result_file)
            
            print(f"正在从 {download_url} 下载对接结果文件...")
            response = requests.get(download_url, stream=True, timeout=60)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_files.append(result_file)
            else:
                failed_files.append({
                    "filename": result_file,
                    "error": f"下载失败，服务器返回: {response.status_code}"
                })
        except Exception as e:
            print(f"文件下载失败 {result_file}: {str(e)}")
            failed_files.append({
                "filename": result_file, 
                "error": str(e)
            })
    
    if failed_files:
        if downloaded_files:
            return {
                "status": "partial",
                "message": f"部分文件下载成功，{len(downloaded_files)}个成功，{len(failed_files)}个失败",
                "downloaded": downloaded_files,
                "failed": failed_files
            }
        else:
            return {
                "status": "error",
                "message": "所有文件下载失败",
                "failed": failed_files
            }
    else:
        return {
            "status": "success",
            "message": f"全部{len(downloaded_files)}个文件成功下载到 {output_dir}",
            "downloaded": downloaded_files
        }

def conformation_evaluation(pred_file, cond_file, dock_mode):
    """执行构象评估计算
    
    Args:
        params: 包含以下字段的字典:
            pred_file: 预测的构象文件路径（pdbqt格式）
            cond_file: 条件蛋白质文件路径（pdb格式）
            dock_mode: 对接模式，可选值为"adgpu"或"vina"
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 评估结果或错误信息}
    """
    # 构建params字典
    params = {
        'pred_file': pred_file,
        'cond_file': cond_file,
        'dock_mode': dock_mode
    }

    print(f"收到构象评估请求，参数: {params}")
    
    # 参数校验
    if not params.get('pred_file'):
        return {"status": "error", "message": "未提供预测构象文件路径"}
        
    if not params.get('cond_file'):
        return {"status": "error", "message": "未提供条件蛋白质文件路径"}
    
    if not params.get('dock_mode'):
        return {"status": "error", "message": "未提供对接模式"}
    
    pred_path = params['pred_file']
    cond_path = params['cond_file']
    dock_mode = params['dock_mode']
    
    # 文件存在性检查
    if not os.path.exists(pred_path):
        return {"status": "error", "message": f"预测构象文件不存在: {pred_path}"}

        
    if not os.path.exists(cond_path):
        return {"status": "error", "message": f"条件蛋白质文件不存在: {cond_path}"}
    
    # 文件格式检查
    if not pred_path.endswith('.pdbqt'):
        return {"status": "error", "message": f"预测构象文件格式错误，必须是.pdbqt格式: {pred_path}"}
    
        
    if not cond_path.endswith('.pdb'):
        return {"status": "error", "message": f"条件蛋白质文件格式错误，必须是.pdb格式: {cond_path}"}
    
    # 对接模式检查
    if dock_mode not in ['adgpu', 'vina']:
        return {"status": "error", "message": f"对接模式错误，必须是'adgpu'或'vina': {dock_mode}"}
    
    # 构建API请求负载
    try:
        with open(pred_path, 'rb') as pred_file, open(cond_path, 'rb') as cond_file:
            files = {
                'pred_file': pred_file,
                'cond_file': cond_file
            }
            
            data = {
                'dock_mode': dock_mode
            }
            
            # 调用Flask API
            print(f"正在调用构象评估API，模式: {dock_mode}...")
            response = requests.post(
                "http://localhost:5000/api/conformation_evaluation",
                files=files,
                data=data,
                timeout=300  # 根据计算时长调整
            )
            
            print(f"API响应: {response.text}")
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success", 
                    "message": f"构象评估计算完成 ({dock_mode}模式)",
                    "result": result
                }
            else:
                return {
                    "status": "error", 
                    "message": f"API返回错误: {response.status_code}", 
                    "response": response.text
                }
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return {"status": "error", "message": f"API调用失败: {str(e)}"}

def download_evaluation_result(result_file, output_path):
    """下载构象评估结果文件
    
    Args:
        params: 包含以下字段的字典:
            result_file: 结果文件名 (如 "posebusters_results.csv")
            output_path: 保存文件的本地路径 (可选，默认为当前目录下的同名文件)
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "message": 操作结果或错误信息, "file_path": 保存的文件路径}
    """
    # 构建params字典
    params = {
        'result_file': result_file,
        'output_path': output_path
    }
    
    print(f"收到评估结果下载请求，参数: {params}")
    
    # 参数校验
    result_file = params.get('result_file')
    if not result_file:
        return {"status": "error", "message": "未提供结果文件名"}
    
    # 获取输出路径，如未提供则使用当前目录和原始文件名
    output_path = params.get('output_path')
    if not output_path:
        output_path = os.path.join(os.getcwd(), result_file)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            return {"status": "error", "message": f"无法创建输出目录: {str(e)}"}
    
    # 构建下载URL
    try:
        download_url = f"http://localhost:5000/api/download/conformation_evaluation/{result_file}"
        
        print(f"正在从 {download_url} 下载评估结果文件...")
        response = requests.get(download_url, stream=True, timeout=60)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "status": "success",
                "message": f"评估结果文件成功下载到 {output_path}",
                "file_path": output_path
            }
        else:
            return {
                "status": "error",
                "message": f"下载失败，服务器返回: {response.status_code}",
                "response": response.text
            }
    except Exception as e:
        print(f"评估结果下载失败: {str(e)}")
        return {"status": "error", "message": f"下载失败: {str(e)}"}




# 为了向外部暴露API，创建带装饰器的版本
@function_tool
def molecule_generation_tool(pdb_file, ref_ligand="A:330", n_samples=1):
    """执行分子生成计算
    
    Args:
        pdb_file: 受体文件绝对路径（必须为.pdb格式）
        ref_ligand: 参考配体信息，可以是"A:330"（默认值，无参考配体）或者SDF文件的绝对路径
        n_samples: 生成样本数量（可选，默认为1）
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 计算结果或错误信息}
    """
    return molecule_generation(pdb_file, ref_ligand, n_samples)

@function_tool
def download_molecule_tool(molecule_name, output_path):
    """下载生成的分子文件
    
    Args:
        molecule_name: 分子文件名 (如 "3rfm_mol.sdf")
        output_path: 保存文件的本地路径 (可选，默认为当前目录下的同名文件)
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "message": 操作结果或错误信息, "file_path": 保存的文件路径}
    """
    return download_molecule(molecule_name, output_path)

@function_tool
def molecular_docking_tool(ligand_sdf, protein_pdb, dock_mode):
    """执行分子对接计算
    
    Args:
        ligand_sdf: 配体文件绝对路径（必须为.sdf格式）
        protein_pdb: 受体文件绝对路径（必须为.pdb格式）
        dock_mode: 对接模式，可选值为"adgpu"或"vina"
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 计算结果或错误信息, "result_files": 结果文件列表}
    """
    return molecular_docking(ligand_sdf, protein_pdb, dock_mode)

@function_tool
def batch_download_docking_results_tool(result_files, output_dir):
    """批量下载分子对接结果文件
    
    Args:
        result_files: 结果文件名列表
        output_dir: 保存文件的目录路径
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "message": 操作结果或错误信息, "downloaded": 成功下载的文件列表, "failed": 下载失败的文件列表}
    """
    return batch_download_docking_results(result_files, output_dir)

@function_tool
def conformation_evaluation_tool(pred_file, cond_file, dock_mode):
    """执行构象评估计算
    
    Args:
        params: 包含以下字段的字典:
            pred_file: 预测的构象文件路径（pdbqt格式）
            cond_file: 条件蛋白质文件路径（pdb格式）
            dock_mode: 对接模式，可选值为"adgpu"或"vina"
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 评估结果或错误信息}
    """
    return conformation_evaluation(pred_file, cond_file, dock_mode)



# 创建一个复合操作助手
complex_agent = Agent(
    name="ComplexAssistant", 
    instructions="""你是一个能够处理复合文件操作请求的助手。你可以：
    1. 执行分子生成操作
    2. 执行分子对接计算
    3. 执行构象评估操作
    
    对于用户的复合请求，请按照正确的顺序执行操作。例如，如果用户要求在新文件夹中创建文件，
    你应该先创建文件夹，然后再在该文件夹中创建文件。
    
    分析用户请求，提取出所有需要执行的操作，然后按正确的顺序调用相应的工具函数。
    
    请确保准确理解用户的路径需求，例如"在1文件夹下创建test.txt"意味着路径应该是"1/test.txt"。

    重要提示：    
    1. 如果用户只需要分子生成，没有提到下载，那么只需要使用molecule_generation_tool这个工具。
    
    2. 如果用户请求执行分子对接计算，使用molecular_docking_tool工具。

    3. 如果用户只需要进行构象评估，则使用conformation_evaluation_tool工具。
    
    """,
    tools=[
        molecule_generation_tool, 
        molecular_docking_tool,
        conformation_evaluation_tool
    ],
    model=deepseek_model
)

# 改进的异步函数，增强了对工具调用的处理
async def chat(agent):
    input_items = []
    
    # 打印欢迎信息和使用提示
    print("\n====== 分子设计工作流助手 ======")
    print("本助手可以帮您完成以下任务：")
    print("1. 分子生成 - 根据受体结构生成小分子")
    print("2. 分子对接 - 使用adgpu或vina模式进行分子对接")
    print("3. 构象评估 - 进行构象合理性评估")
    print("\n您可以通过自然语言描述需要执行的任务，下面是一些示例：")
    
    # 定义示例提示，并用颜色高亮显示
    examples = [
        "请使用/home/zhangfn/workflow/3rfm.pdb生成2个分子",
        "请将生成的分子3rfm_mol.sdf下载到/home/zhangfn/workflow/3rfm_mol.sdf",
        "请执行vina模式的分子对接，使用/home/zhangfn/workflow/3rfm_mol.sdf作为配体，/home/zhangfn/workflow/3rfm.pdb作为受体",
        "请下载对接结果到/home/zhangfn/workflow目录",
        "请使用/home/zhangfn/test_file/3rfm_ligand_0_vina.pdbqt作为pred_file，/home/zhangfn/workflow/3rfm.pdb作为cond_file，vina作为dock_mode进行构象评估",
    ]
    
    # 打印带颜色的示例
    for i, example in enumerate(examples):
        print(f"\033[96m示例{i+1}: {example}\033[0m")
    
    print("\n" + "="*35)
    
    while True:
        try:
            input_items = []
            print("\n您可以输入需要执行的任务，或输入'help'查看帮助信息：")
            user_input = input("\033[95m💬 请输入您的指令：\033[0m ")
            
            # 处理特殊命令
            if user_input.lower() in ["exit", "quit"]:
                print("\033[92m✅ 对话已结束\033[0m")
                return
            elif user_input.lower() == "help":
                print("\n\033[93m==== 帮助信息 ====\033[0m")
                print("您可以执行以下操作：")
                print("- 分子生成: 指定受体PDB文件路径和可选的参考配体")
                print("- 分子下载: 指定分子名称和保存路径")
                print("- 分子对接: 指定配体SDF文件、受体PDB文件和对接模式(adgpu/vina)")
                print("- 对接结果下载: 指定结果文件和保存目录")
                print("- 构象评估: 指定预测配体、真实配体和受体蛋白")
                print("\n示例命令：")
                for i, example in enumerate(examples):
                    print(f"\033[96m示例{i+1}: {example}\033[0m")
                continue
                
            # # 只保留最近一轮对话
            # if len(input_items) > 2:  # 仅保留一轮对话（用户+助手）
            #     input_items = input_items[-2:]   

            # 处理正常的用户输入
            # input_items.append({"content": user_input, "role": "user"})
            input_items = [{"content": user_input, "role": "user"}]
            
            # 显示处理中提示
            print("\033[93m正在处理您的请求...\033[0m")
            
            # 运行智能体并处理工具调用
            result = await Runner.run(agent, input_items)
            
            # # 显示结果
            # if hasattr(result, 'final_output') and result.final_output:
            #     print("\n\033[92m===== 执行结果 =====\033[0m")
            #     display(Markdown(result.final_output))
            
            # # 如果有回复，则将其添加到输入列表中
            # if hasattr(result, 'to_input_list'):
            #     # input_items = result.to_input_list()
            #     # 清理旧的 tool_call 相关内容，只保留 user/assistant
            #     input_items = [
            #         item for item in result.to_input_list()
            #         if item.get("role") in ["user", "assistant"]
            #     ]


            #     # 可选：显示调试信息（可以注释掉或设置debug标志）
            #     debug = True  # 设置为True开启调试输出
            #     if debug:
            #         print("\n\033[93m===== 调试信息 =====\033[0m")
            #         print("处理后的输入列表:")
            #         for item in input_items:
            #             if item.get('role') == 'assistant':
            #                 print(f"Assistant: {item.get('content', '')[:100]}...")
            #             elif item.get('type') == 'function_call':
            #                 print(f"Function Call: {item.get('name')}")
            #             elif item.get('type') == 'function_call_output':
            #                 print(f"Function Output: {item.get('output', '')[:100]}...")
            #             else:
            #                 print(f"{item.get('role', 'unknown')}: {item.get('content', '')[:100]}...")
                    
            #         print("*****************************")
            #         if hasattr(result, 'final_output'):
            #             print(result.final_output[:200])
            #             if len(result.final_output) > 200:
            #                 print("...")
            
        except KeyboardInterrupt:
            print("\n\033[92m✅ 操作已中断，对话结束\033[0m")
            return
        except Exception as e:
            print(f"\n\033[91m发生错误：{e}\033[0m")
            import traceback
            traceback.print_exc()
            print("\n\033[93m您可以尝试重新输入或使用不同的表达方式\033[0m")

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
    # # 运行异步函数
    # try:
    #     asyncio.run(main())
    # except Exception as e:
    #     print(f"程序运行时发生错误：{e}")
    # finally:
    #     # 确保程序退出
    #     print("程序结束")
    #     os._exit(0)

    asyncio.run(main())
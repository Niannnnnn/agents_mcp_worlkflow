import json
import os
import requests
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("分子对接服务器启动中...")

# 初始化 MCP 服务器
mcp = FastMCP("MoleculeDockingServer")

DEFAULT_LIGAND_DIR = "/home/zhangfn/workflow/downloads"
DEFAULT_PROTEIN_DIR = "/home/zhangfn/workflow/uploads"

def find_first_file_with_ext(directory: str, extension: str) -> str:
    """在目录中找到第一个指定扩展名的文件"""
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)
    raise FileNotFoundError(f"{directory} 中没有找到以 {extension} 结尾的文件")

@mcp.tool()
def molecular_docking(ligand_sdf=None, protein_pdb=None, dock_mode="adgpu"):
    """执行分子对接计算
    
    Args:
        ligand_sdf: 配体文件绝对路径（必须为.sdf格式）
        protein_pdb: 受体文件绝对路径（必须为.pdb格式）
        dock_mode: 对接模式，可选值为"adgpu"或"vina"
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 计算结果或错误信息, "result_files": 结果文件列表}
    """
    # 如果用户没有提供ligand_sdf和protein_pdb参数，使用默认值
    if not ligand_sdf:
        try:
            ligand_sdf = find_first_file_with_ext(DEFAULT_LIGAND_DIR, '.sdf')
            print(f"使用ligand_sdf: {ligand_sdf}")
        except Exception as e:
            return {"status": "error", "message": f"未提供配体SDF文件，且默认路径中未找到合适文件：{str(e)}"}
    if not protein_pdb:
        try:
            protein_pdb = find_first_file_with_ext(DEFAULT_PROTEIN_DIR, '.pdb')
            print(f"使用protein_pdb: {protein_pdb}")
        except Exception as e:
            return {"status": "error", "message": f"未提供受体PDB文件，且默认路径中未找到合适文件：{str(e)}"}

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

def main():
    logging.info("分子对接服务器启动，使用stdio通信...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
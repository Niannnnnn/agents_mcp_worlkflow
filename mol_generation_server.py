import json
import os
import requests
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("分子生成服务器启动中...")

# 初始化 MCP 服务器
mcp = FastMCP("MoleculeGenerationServer")

@mcp.tool()
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

def main():
    logging.info("分子生成服务器启动，使用stdio通信...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
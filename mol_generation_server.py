import json
import os
import requests
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug("分子生成服务器启动中...")

# 初始化 MCP 服务器
mcp = FastMCP("MoleculeGenerationServer")

WORKING_DIR = Path("/home/zhangfn/workflow")
REF_FOLDER = WORKING_DIR / "ref"
UPLOAD_FOLDER = WORKING_DIR / "uploads"

@mcp.tool()
def molecule_generation(pdb_file, ref_ligand="A:330", n_samples=1):
    """执行分子生成计算

    Args:
        pdb_file: 受体文件绝对路径（必须为.pdb格式）、也可能是"uploaded_pdb"字段
        ref_ligand: 参考配体信息，可以是"A:330"（默认值，无参考配体）、"best_ref_ligand_sdf"（使用REF_FOLDER中的最佳参考配体）或者SDF文件的绝对路径
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

    # 参数校验和文件路径处理
    pdb_path = params['pdb_file']

    if pdb_path == "uploaded_pdb":
        try:
            pdb_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdb")]
            if not pdb_files:
                return {"status": "error", "message": "UPLOAD_FOLDER 中未找到 .pdb 文件"}
            elif len(pdb_files) > 1:
                return {"status": "error", "message": "UPLOAD_FOLDER 中存在多个 .pdb 文件，请手动指定"}
            else:
                pdb_path = UPLOAD_FOLDER / pdb_files[0]
                print(f"使用上传的 .pdb 文件: {pdb_path}")
        except Exception as e:
            return {"status": "error", "message": f"读取上传的 .pdb 文件失败: {str(e)}"}
    else:
        if not os.path.exists(pdb_path):
            return {"status": "error", "message": f"PDB文件不存在: {pdb_path}"}
        if not pdb_path.endswith('.pdb'):
            return {"status": "error", "message": f"文件格式错误，必须是.pdb格式: {pdb_path}"}

    # 处理参考配体
    ref_ligand = params.get('ref_ligand', 'A:330')

    # 处理best_ref_ligand_sdf特殊值
    if ref_ligand == 'best_ref_ligand_sdf':
        try:
            # 寻找REF_FOLDER中的SDF文件
            sdf_files = [f for f in os.listdir(REF_FOLDER) if f.endswith('.sdf')]
            if not sdf_files:
                print(f"警告: REF_FOLDER中没有找到SDF文件，将使用默认参考配体")
                ref_ligand = 'A:330'
            else:
                # 使用找到的第一个SDF文件
                ref_ligand = os.path.join(REF_FOLDER, sdf_files[0])
                print(f"找到最佳参考配体: {ref_ligand}")
        except Exception as e:
            print(f"查找最佳参考配体时出错: {str(e)}，将使用默认参考配体")
            ref_ligand = 'A:330'

    # 正常校验参考配体
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

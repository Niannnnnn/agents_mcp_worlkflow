import json
import os
import requests
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
import re
import glob

import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("分子构象评估服务器启动中...")

# 初始化 MCP 服务器
mcp = FastMCP("MoleculeEvalServer")

@mcp.tool()
def conformation_evaluation(pred_file=None, cond_file=None, dock_mode="vina"):
    """执行构象评估计算"""

    def get_default_pred_files():
        pdbqt_files = glob.glob("/home/zhangfn/workflow/downloads/*.pdbqt")
        def extract_sort_key(file_path):
            filename = os.path.basename(file_path)
            match = re.search(r'_([0-9]+)_', filename)
            return int(match.group(1)) if match else float('inf')
        return sorted(pdbqt_files, key=extract_sort_key)

    def get_default_cond_file():
        pdb_files = glob.glob("/home/zhangfn/workflow/uploads/*.pdb")
        return pdb_files[0] if pdb_files else None

    # 设置默认参数
    if not pred_file:
        pred_file = get_default_pred_files()
    if not cond_file:
        cond_file = get_default_cond_file()

    print(f"收到构象评估请求，pred_file: {pred_file}, cond_file: {cond_file}, dock_mode: {dock_mode}")

    if not pred_file:
        return {"status": "error", "message": "未能找到任何默认预测构象文件（.pdbqt）"}
    if not cond_file:
        return {"status": "error", "message": "未能找到默认的条件蛋白质文件（.pdb）"}

    # 统一处理成列表
    if isinstance(pred_file, str):
        pred_file = [pred_file]

    if not os.path.exists(cond_file):
        return {"status": "error", "message": f"条件蛋白质文件不存在: {cond_file}"}
    if not cond_file.endswith('.pdb'):
        return {"status": "error", "message": f"条件蛋白质文件格式错误，必须是.pdb格式: {cond_file}"}
    if dock_mode not in ['adgpu', 'vina']:
        return {"status": "error", "message": f"对接模式错误，必须是'adgpu'或'vina': {dock_mode}"}

    results = []
    for pred_path in pred_file:
        if not os.path.exists(pred_path):
            results.append({"file": pred_path, "status": "error", "message": f"预测构象文件不存在"})
            continue
        if not pred_path.endswith('.pdbqt'):
            results.append({"file": pred_path, "status": "error", "message": f"文件格式错误，不是.pdbqt: {pred_path}"})
            continue

        try:
            with open(pred_path, 'rb') as pf, open(cond_file, 'rb') as cf:
                files = {
                    'pred_file': pf,
                    'cond_file': cf
                }
                data = {
                    'dock_mode': dock_mode
                }

                print(f"调用API进行评估，文件: {pred_path}")
                response = requests.post(
                    "http://localhost:5000/api/conformation_evaluation",
                    files=files,
                    data=data,
                    timeout=300
                )

                if response.status_code == 200:
                    results.append({
                        "file": pred_path,
                        "status": "success",
                        "result": response.json()
                    })
                else:
                    results.append({
                        "file": pred_path,
                        "status": "error",
                        "message": f"API错误: {response.status_code}",
                        "response": response.text
                    })
        except Exception as e:
            results.append({
                "file": pred_path,
                "status": "error",
                "message": f"API调用失败: {str(e)}"
            })

    return {
        "status": "success",
        "message": f"共处理 {len(results)} 个文件",
        "results": results
    }



def main():
    logging.info("分子构象评估服务器启动，使用stdio通信...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
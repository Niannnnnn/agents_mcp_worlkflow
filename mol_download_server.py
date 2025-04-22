import json
import os
import requests
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("分子下载服务器启动中...")

# 初始化 MCP 服务器
mcp = FastMCP("MoleculeDownloadingServer")

@mcp.tool()
def download_all_outputs(output_path=None):
    """下载整个 download 目录的所有文件，并解压到指定目录
    
    Args:
        output_path: 解压后保存文件的本地目录（可选，默认当前目录 ./downloaded_outputs）
    
    Returns:
        dict: 包含状态、提示信息和文件保存目录
    """
    import requests, zipfile, io, os

    print(f"收到下载所有输出文件的请求，output_path={output_path}")
    
    # 设置默认保存路径
    if not output_path:
        output_path = os.path.join(os.getcwd(), "downloaded_outputs")
    
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except Exception as e:
            return {"status": "error", "message": f"无法创建输出目录: {str(e)}"}

    download_url = "http://localhost:5000/api/download_all"
    print(f"正在从 {download_url} 下载所有文件...")

    try:
        response = requests.get(download_url, stream=True, timeout=300)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(output_path)

            return {
                "status": "success",
                "message": f"所有文件已成功下载并解压到 {output_path}",
                "output_path": output_path
            }
        else:
            return {
                "status": "error",
                "message": f"下载失败，服务器返回状态码: {response.status_code}",
                "response": response.text
            }
    except Exception as e:
        print(f"下载或解压失败: {str(e)}")
        return {"status": "error", "message": f"下载失败: {str(e)}"}

def main():
    logging.info("分子下载服务器启动，使用stdio通信...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
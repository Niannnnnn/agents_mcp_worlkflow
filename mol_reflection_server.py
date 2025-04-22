import json
import os
import requests
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("结果反馈服务器启动中...")

# 初始化 MCP 服务器
mcp = FastMCP("MolReflectionServer")

@mcp.tool()
def molecule_reflection():
    """评估分子对接结合能和构象质量
    
    评估两个指标：
    1. 对接结合能是否小于-5
    2. Posebusters构象评估的各项指标是否均为True
    
    Returns:
        dict: 包含评估结果的状态和详细信息
    """
    import requests
    
    print("正在调用结果反馈API...")
    
    reflection_url = "http://localhost:5000/api/reflection"
    
    try:
        response = requests.post(reflection_url, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # 处理结果
            all_results = result.get("results", [])
            passed_count = sum(1 for item in all_results if item.get("overall_pass") == "YES")
            total_count = len(all_results)
            
            # 构建返回消息
            summary = {
                "status": "success",
                "message": all_results
            }
            
            # 打印简要结果
            print(f"评估完成: {passed_count}/{total_count} 个分子通过所有评估指标")
            for item in all_results:
                filename = item.get("filename", "未知文件")
                result_str = "通过" if item.get("overall_pass") == "YES" else "未通过"
                print(f"- {filename}: {result_str}")
                
                # 打印未通过原因
                if item.get("overall_pass") == "NO":
                    if not item.get("binding_energy_pass"):
                        energy = item.get("binding_energy", "未知")
                        print(f"  - 对接结合能 ({energy}) 未小于 -5")
                    if not item.get("posebusters_pass"):
                        print(f"  - 构象评估未通过所有指标")
            
            return summary
        else:
            error_msg = f"API调用失败，状态码: {response.status_code}"
            print(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "response": response.text
            }
    except Exception as e:
        error_msg = f"评估过程出错: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

def main():
    logging.info("结果反馈服务器启动，使用stdio通信...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
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
def molecule_generation(params: Dict[str, Any]) -> Dict:
    """执行分子生成计算
    
    Args:
        params: 包含以下字段的字典:
            pdb_file: 受体文件绝对路径（必须为.pdb格式）
            ref_ligand: 参考配体信息，可以是"A:330"（默认值，无参考配体）或者SDF文件的绝对路径
            n_samples: 生成样本数量（可选，默认为2）
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 计算结果或错误信息}
    """
    # 显式定义 inputSchema
    molecule_generation.inputSchema = {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {
                    "pdb_file": {
                        "type": "string",
                        "description": "受体文件绝对路径（必须为.pdb格式）"
                    },
                    "ref_ligand": {
                        "type": "string",
                        "description": "参考配体信息，可以是\"A:330\"（默认值，无参考配体）或者SDF文件的绝对路径"
                    },
                    "n_samples": {
                        "type": "integer",
                        "description": "生成样本数量",
                        "default": 2
                    }
                },
                "required": ["pdb_file"]
            }
        },
        "required": ["params"]
    }
    
    logging.debug(f"收到分子生成请求，参数: {params}")
    
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
    
    n_samples = params.get('n_samples', 2)
    
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
        logging.debug(f"正在调用分子生成API...")
        response = requests.post(
            "http://localhost:5000/api/molecule_generation",
            files=files,
            data=data,
            timeout=300
        )
            
        logging.debug(f"API响应: {response.text}")
        if response.status_code == 200:
            result = response.json()
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
        logging.error(f"API调用失败: {str(e)}")
        return {"status": "error", "message": f"API调用失败: {str(e)}"}

@mcp.tool()
def download_molecule(params: Dict[str, Any]) -> Dict:
    """下载生成的分子文件
    
    Args:
        params: 包含以下字段的字典:
            molecule_name: 分子文件名 (如 "3rfm_mol.sdf")
            output_path: 保存文件的本地路径 (可选，默认为当前目录下的同名文件)
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "message": 操作结果或错误信息, "file_path": 保存的文件路径}
    """
    # 显式定义 inputSchema
    download_molecule.inputSchema = {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {
                    "molecule_name": {
                        "type": "string",
                        "description": "要下载的分子文件名 (如 '3rfm_mol.sdf')"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "保存文件的本地路径 (可选，默认为当前目录下的同名文件)"
                    }
                },
                "required": ["molecule_name"]
            }
        },
        "required": ["params"]
    }
    
    logging.debug(f"收到分子下载请求，参数: {params}")
    
    # 参数校验
    molecule_name = params.get('molecule_name')
    if not molecule_name:
        return {"status": "error", "message": "未提供分子文件名"}
    
    # 获取输出路径，如未提供则使用当前目录和原始文件名
    output_path = params.get('output_path')
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
        # 从文件名中提取PDB ID (假设命名格式为 "xxxx_mol.sdf")
        pdb_id = molecule_name.split('_')[0] if '_' in molecule_name else molecule_name.split('.')[0]
        download_url = f"http://localhost:5000/api/download/molecule_generation/{molecule_name}"
        
        logging.debug(f"正在从 {download_url} 下载分子文件...")
        response = requests.get(download_url, stream=True, timeout=60)
        
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
        logging.error(f"分子下载失败: {str(e)}")
        return {"status": "error", "message": f"下载失败: {str(e)}"}

@mcp.tool()
def molecular_docking(params: Dict[str, Any]) -> Dict:
    """执行分子对接计算
    
    Args:
        params: 包含以下字段的字典:
            ligand_sdf: 配体文件绝对路径（必须为.sdf格式）
            protein_pdb: 受体文件绝对路径（必须为.pdb格式）
            dock_mode: 对接模式，可选值为"adgpu"或"vina"
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 计算结果或错误信息}
    """
    # 显式定义 inputSchema
    molecular_docking.inputSchema = {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {
                    "ligand_sdf": {
                        "type": "string",
                        "description": "配体文件绝对路径（必须为.sdf格式）"
                    },
                    "protein_pdb": {
                        "type": "string",
                        "description": "受体文件绝对路径（必须为.pdb格式）"
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
    
    logging.debug(f"收到分子对接请求，参数: {params}")
    
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
                'ligand_sdf': ligand_file,
                'protein_pdb': protein_file
            }
            
            data = {
                'dock_mode': dock_mode
            }
            
            # 调用Flask API
            logging.debug(f"正在调用分子对接API，模式: {dock_mode}...")
            response = requests.post(
                "http://localhost:5000/api/molecular_docking",
                files=files,
                data=data,
                timeout=600  # 根据计算时长调整，分子对接可能需要更长时间
            )
            
            logging.debug(f"API响应: {response.text}")
            if response.status_code == 200:
                result = response.json()

                result_files = result.get('result_files', [])
                
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
        logging.error(f"API调用失败: {str(e)}")
        return {"status": "error", "message": f"API调用失败: {str(e)}"}

@mcp.tool()
def download_docking_result(params: Dict[str, Any]) -> Dict:
    """下载分子对接结果文件
    
    Args:
        params: 包含以下字段的字典:
            result_file: 结果文件名 (如 "3rfm_ligand_0_1.pdbqt" 或 "3rfm_ligand_2.pdbqt")
            output_path: 保存文件的本地路径 (可选，默认为当前目录下的同名文件)
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "message": 操作结果或错误信息, "file_path": 保存的文件路径}
    """
    # 显式定义 inputSchema
    download_docking_result.inputSchema = {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {
                    "result_file": {
                        "type": "string",
                        "description": "要下载的对接结果文件名 (如 '3rfm_ligand_0_1.pdbqt' 或 '3rfm_ligand_2.pdbqt')"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "保存文件的本地路径 (可选，默认为当前目录下的同名文件)"
                    }
                },
                "required": ["result_file"]
            }
        },
        "required": ["params"]
    }
    
    logging.debug(f"收到对接结果下载请求，参数: {params}")
    
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
        download_url = f"http://localhost:5000/api/download/molecular_docking/{result_file}"
        
        logging.debug(f"正在从 {download_url} 下载对接结果文件...")
        response = requests.get(download_url, stream=True, timeout=60)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "status": "success",
                "message": f"对接结果文件成功下载到 {output_path}",
                "file_path": output_path
            }
        else:
            return {
                "status": "error",
                "message": f"下载失败，服务器返回: {response.status_code}",
                "response": response.text
            }
    except Exception as e:
        logging.error(f"对接结果下载失败: {str(e)}")
        return {"status": "error", "message": f"下载失败: {str(e)}"}


@mcp.tool()
def batch_download_docking_results(params: Dict[str, Any]) -> Dict:
    """批量下载分子对接结果文件
    
    Args:
        params: 包含以下字段的字典:
            result_files: 结果文件名列表
            output_dir: 保存文件的目录路径
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "message": 操作结果或错误信息, "downloaded": 成功下载的文件列表, "failed": 下载失败的文件列表}
    """
    # 显式定义 inputSchema
    batch_download_docking_results.inputSchema = {
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
    
    logging.debug(f"收到批量对接结果下载请求，参数: {params}")
    
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
            
            logging.debug(f"正在从 {download_url} 下载对接结果文件...")
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
            logging.error(f"文件下载失败 {result_file}: {str(e)}")
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


@mcp.tool()
def conformation_evaluation(params: Dict[str, Any]) -> Dict:
    """执行构象评估计算
    
    Args:
        params: 包含以下字段的字典:
            pred_file: 预测的构象文件路径（pdbqt格式）
            true_file: 参考的真实构象文件路径（pdbqt格式）
            cond_file: 条件蛋白质文件路径（pdb格式）
            dock_mode: 对接模式，可选值为"adgpu"或"vina"
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "result": 评估结果或错误信息}
    """
    # 显式定义 inputSchema
    conformation_evaluation.inputSchema = {
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
    
    logging.debug(f"收到构象评估请求，参数: {params}")
    
    # 参数校验
    if not params.get('pred_file'):
        return {"status": "error", "message": "未提供预测构象文件路径"}
    
    if not params.get('true_file'):
        return {"status": "error", "message": "未提供真实构象文件路径"}
        
    if not params.get('cond_file'):
        return {"status": "error", "message": "未提供条件蛋白质文件路径"}
    
    if not params.get('dock_mode'):
        return {"status": "error", "message": "未提供对接模式"}
    
    pred_path = params['pred_file']
    true_path = params['true_file']
    cond_path = params['cond_file']
    dock_mode = params['dock_mode']
    
    # 文件存在性检查
    if not os.path.exists(pred_path):
        return {"status": "error", "message": f"预测构象文件不存在: {pred_path}"}
    
    if not os.path.exists(true_path):
        return {"status": "error", "message": f"真实构象文件不存在: {true_path}"}
        
    if not os.path.exists(cond_path):
        return {"status": "error", "message": f"条件蛋白质文件不存在: {cond_path}"}
    
    # 文件格式检查
    if not pred_path.endswith('.pdbqt'):
        return {"status": "error", "message": f"预测构象文件格式错误，必须是.pdbqt格式: {pred_path}"}
    
    if not true_path.endswith('.pdbqt'):
        return {"status": "error", "message": f"真实构象文件格式错误，必须是.pdbqt格式: {true_path}"}
        
    if not cond_path.endswith('.pdb'):
        return {"status": "error", "message": f"条件蛋白质文件格式错误，必须是.pdb格式: {cond_path}"}
    
    # 对接模式检查
    if dock_mode not in ['adgpu', 'vina']:
        return {"status": "error", "message": f"对接模式错误，必须是'adgpu'或'vina': {dock_mode}"}
    
    # 构建API请求负载
    try:
        with open(pred_path, 'rb') as pred_file, open(true_path, 'rb') as true_file, open(cond_path, 'rb') as cond_file:
            files = {
                'pred_file': pred_file,
                'true_file': true_file,
                'cond_file': cond_file
            }
            
            data = {
                'dock_mode': dock_mode
            }
            
            # 调用Flask API
            logging.debug(f"正在调用构象评估API，模式: {dock_mode}...")
            response = requests.post(
                "http://localhost:5000/api/conformation_evaluation",
                files=files,
                data=data,
                timeout=300  # 根据计算时长调整
            )
            
            logging.debug(f"API响应: {response.text}")
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
        logging.error(f"API调用失败: {str(e)}")
        return {"status": "error", "message": f"API调用失败: {str(e)}"}

@mcp.tool()
def download_evaluation_result(params: Dict[str, Any]) -> Dict:
    """下载构象评估结果文件
    
    Args:
        params: 包含以下字段的字典:
            result_file: 结果文件名 (如 "posebusters_results.csv")
            output_path: 保存文件的本地路径 (可选，默认为当前目录下的同名文件)
    
    Returns:
        包含状态和结果的字典: {"status": "success/failure", "message": 操作结果或错误信息, "file_path": 保存的文件路径}
    """
    # 显式定义 inputSchema
    download_evaluation_result.inputSchema = {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {
                    "result_file": {
                        "type": "string",
                        "description": "要下载的评估结果文件名 (如 'posebusters_results.csv')"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "保存文件的本地路径 (可选，默认为当前目录下的同名文件)"
                    }
                },
                "required": ["result_file"]
            }
        },
        "required": ["params"]
    }
    
    logging.debug(f"收到评估结果下载请求，参数: {params}")
    
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
        
        logging.debug(f"正在从 {download_url} 下载评估结果文件...")
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
        logging.error(f"评估结果下载失败: {str(e)}")
        return {"status": "error", "message": f"下载失败: {str(e)}"}




def main():
    logging.info("分子生成服务器启动，使用stdio通信...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
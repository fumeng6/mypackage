import numpy as np
import re
import pickle
import os

def get_time_a(poel_file: str):
    """
    读取poel序列中时间序列的函数

    Parameters:\n
    poel_file：poel程序输出的结果文件（包含文件名的完整路径）

    Returns:\n
    time_a：一组以年为单位的时间序列
    """
    # 获取poel输出的结果
    with open(poel_file, "r") as f:
        lines = f.readlines()
        del lines[0:1]

    # 将包含时间的文本信息转换成数组
    lines = np.array([[float(val) for val in line.split()] for line in lines])
    # 存为时间数组
    time = lines[:, 0]
    # 将poel中以秒为单位的时间转换成以年为单位的时间
    time_a = [round(x / (365.25 * 24 * 60 * 60), 2) for x in time]

    return time_a

def get_poel(poel_file: str):
    """
    读取poel结果文件中计算的孔隙压力、应变、位移等结果值的函数

    Parameters:\n
    poel_file：poel程序输出的结果文件（包含文件名的完整路径）

    Returns:\n
    poel：一个numpy数组，包含从文件中读取到的结果值
    """
    # 获取poel输出的结果
    with open(poel_file, "r") as f:
        lines = f.readlines()
        del lines[0:1]

    # 将包含时间及对应数据的文本信息转换成数组
    lines = np.array([[float(val) for val in line.split()] for line in lines])
    # 存为结果数组,去掉了时间列
    poel = lines[:, 1:]
    return poel

def read_poel(poel_file: str):
    """
    读取poel结果文件中计算的孔隙压力、应变、位移等结果值的函数

    Parameters:\n
    poel_file：poel程序输出的结果文件（包含文件名的完整路径）

    Returns:\n
    poel：一个字典，包含从poel程序输出的结果文件中读取出来的头段信息，时间信息，及结果值
    """
    with open(poel_file, "r") as f:
        lines = f.readlines()
    
    # 创建储存信息的字典poel
    poel = {}
    # 将包含时间及对应数据的文本信息转换成数组
    new_lines = np.array([[float(val) for val in line.split()] for line in lines[1:]])
    # 将结果值存入字典
    poel['value'] =new_lines[:,1:]
    # 将时间信息存入字典
    poel['time'] =new_lines[:,0]

    return poel

def get_time(comsol_file: str):
    """
    读取comsol导出数据中的时间序列的函数

    Parameters:\n
    comsol_file：comsol程序导出的结果文件（包含文件名的完整路径）

    Returns:\n
    time：一个列表，包含从文件中读取到的时间信息
    """
    # 读取文件
    with open(comsol_file, "r") as f:
        lines = f.readlines()
    weidu = int(lines[3].split()[-1])
    time = lines[8].split("\t")
    del time[0:weidu]
    # 用正则表达式匹配文本信息中的时间数字
    time = [float(s.split("=")[-1]) for s in time]

    return time

def get_comsol(comsol_file: str):
    """
    读取comsol软件导出数据中结果值的函数

    Parameters:\n
    comsol_file：comsol程序导出的结果文件（包含文件名的完整路径）

    Returns:\n
    comsol：一个numpy数组，包含从文件中读取到的结果值
    """
    # 获取comsol导出的结果
    with open(comsol_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 获取模型的维度
    weidu = int(lines[3].split()[-1])
    # 对匹配出的内容进行处理
    comsol = lines[9].split()
    del comsol[0:weidu]
    comsol[-1] = comsol[-1].strip("\n")
    comsol = np.array(list(map(float, comsol)))

    return comsol

def readconfig(poel_config_filename: str):
    """
    读取poel配置文件参数的函数

    Parameters:\n
    poel_config_filename：poel程序使用的 .inp 配置文件（包含文件名的完整路径）

    Returns:\n
    config：一个字典，包含所有参数值,键为参数名，值为参数值

    lines：一个字符串列表，包含配置文件中的每一行的字符串

    index：一个字典，包含每一个参数在参数文件中的行索引值，键为参数名，值为行索引
    """
    # 获取poel配置文件中的参数值
    with open(poel_config_filename, "r") as f:
        lines = f.readlines()

    # 构建正则表达式以匹配文件中参数所在行
    patterns = {
        "source_top_bottom_depth": r".*\|dble:\s+s_top_depth,\s+s_bottom_depth;",
        "source_radius": r".*\|dble:\s+s_radius;",
        "source_type": r".*\|int:\s+sw_source_type;",
        "injection_data_lines": r".*\|int:\s+no_data_lines;",
        "p5_end": r"^\d+\s+[\d.]+\s+[\d.-]+\s*$",
        "receiver_depth_sampling": r".*\|int:\s+sw_receiver_depth_sampling;",
        "depth_samples_num": r".*\|int:\s+no_depths;",
        "depth_start_end": r".*\|dble:\s+zr_1,zr_n;\s+or\s+zr_1,zr_2,...,zr_n;",
        "receiver_distance_sampling": r".*\|int:\s+sw_receiver_distance_sampling;",
        "distances_samples_num": r".*\|int:\s+no_distances;",
        "distances_start_end": r".*\|dble:\s+r_1,r_n;\s+or\s+r_1,r_2,...,r_n;",
        "time_window": r".*\|dble:\s+time_window;",
        "time_samples_num": r".*\|int:\s+no_time_samples;",
        "integral_accuracy": r".*\|dble:\s+accuracy;",
        "displacement_time_series": r".*\|int:\s+sw_t_files\(1-3\);",
        "displacement_time_series_file": r".*\|char: t_files\(1-3\);$",
        "strain_tensor_time_series": r".*\|int: sw_t_files\(3-7\);",
        "strain_tensor_time_series_file": r".*\|char:\s+t_files\(3-7\);",
        "pore_pressure_time_series": r".*\|int: sw_t_files\(8-10\);",
        "pore_pressure_time_series_file": r".*\|char:\s+t_files\(8-10\);",
        "snapshots_num": r".*\|int:\s+no_sn;",
        "snapshots_time_file": r".*\|dable:\s+sn_time\(i\),sn_file\(i\), i=1,2,...$",
        "boundary_conditions": r".*\|int:\s+isurfcon$",
        "model_lines": r".*\|int:\s+no_model_lines;$",
        "key_params": r"^\s+\d+\s+\d+(\.\d+)?\s+\d+(\.\d+)?([Ee][+-]?\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?$",
    }

    # 创建参数索引值字典
    index = {}
    # 初始化注水速率随时间变化表的开始行与结束行
    p5_start, p5_end = None, None

    # 开始匹配文件中的参数行
    for line_num, line in enumerate(lines):
        for key, pattern in patterns.items():
            match = re.match(pattern, line)
            if match:
                index[key] = line_num
                if key == "p5_end":
                    if p5_start is None:
                        p5_start = line_num
                        index["p5_start"] = p5_start
                    p5_end = line_num
                break

    # 创建参数名及其参数值的字典
    config = {}

    # 根据索引值输出配置文件中各参数的值
    config["source_top"] = lines[index["source_top_bottom_depth"]].split()[0]
    config["source_bottom"] = lines[index["source_top_bottom_depth"]].split()[1]
    config["source_radius"] = lines[index["source_radius"]].split()[0]
    config["source_type"] = lines[index["source_type"]].split()[0]
    config["injection_data_lines"] = lines[index["injection_data_lines"]].split()[0]
    config["injection_start_end"] = lines[index["p5_start"] : index["p5_end"] + 1]
    config["receiver_depth_sampling"] = lines[index["receiver_depth_sampling"]].split()[
        0
    ]
    config["depth_samples_num"] = lines[index["depth_samples_num"]].split()[0]
    config["depth_start"] = lines[index["depth_start_end"]].split()[0]
    config["depth_end"] = lines[index["depth_start_end"]].split()[1]
    config["receiver_distance_sampling"] = lines[
        index["receiver_distance_sampling"]
    ].split()[0]
    config["distances_samples_num"] = lines[index["distances_samples_num"]].split()[0]
    config["distances_start"] = lines[index["distances_start_end"]].split()[0]
    config["distances_end"] = lines[index["distances_start_end"]].split()[1]
    config["time_window"] = lines[index["time_window"]].split()[0]
    config["time_samples_num"] = lines[index["time_samples_num"]].split()[0]
    config["integral_accuracy"] = lines[index["integral_accuracy"]].split()[0]
    config["displacement_output_switch_z"] = lines[
        index["displacement_time_series"]
    ].split()[0]
    config["displacement_output_switch_r"] = lines[
        index["displacement_time_series"]
    ].split()[1]
    config["displacement_output_z"] = lines[
        index["displacement_time_series_file"]
    ].split()[0]
    config["displacement_output_r"] = lines[
        index["displacement_time_series_file"]
    ].split()[1]
    config["strain_output_switch_zz"] = lines[
        index["strain_tensor_time_series"]
    ].split()[0]
    config["strain_output_switch_rr"] = lines[
        index["strain_tensor_time_series"]
    ].split()[1]
    config["strain_output_switch_tt"] = lines[
        index["strain_tensor_time_series"]
    ].split()[2]
    config["strain_output_switch_zr"] = lines[
        index["strain_tensor_time_series"]
    ].split()[3]
    config["tlt_output_switch"] = lines[index["strain_tensor_time_series"]].split()[4]
    config["strain_output_zz"] = lines[index["strain_tensor_time_series_file"]].split()[
        0
    ]
    config["strain_output_rr"] = lines[index["strain_tensor_time_series_file"]].split()[
        1
    ]
    config["strain_output_tt"] = lines[index["strain_tensor_time_series_file"]].split()[
        2
    ]
    config["strain_output_zr"] = lines[index["strain_tensor_time_series_file"]].split()[
        3
    ]
    config["tlt_output"] = lines[index["strain_tensor_time_series_file"]].split()[4]
    config["pore_pressure_output_switch"] = lines[
        index["pore_pressure_time_series"]
    ].split()[0]
    config["darcy_output_switch_z"] = lines[index["pore_pressure_time_series"]].split()[
        1
    ]
    config["darcy_output_switch_r"] = lines[index["pore_pressure_time_series"]].split()[
        2
    ]
    config["pore_pressure_output"] = lines[
        index["pore_pressure_time_series_file"]
    ].split()[0]
    config["snapshots_num"] = lines[index["snapshots_num"]].split()[0]
    config["snapshots_output_time"] = lines[index["snapshots_time_file"]].split()[0]
    config["snapshots_output_file"] = lines[index["snapshots_time_file"]].split()[0]
    config["boundary_conditions"] = lines[index["boundary_conditions"]].split()[0]
    config["model_lines"] = lines[index["model_lines"]].split()[0]
    config["row"] = lines[index["key_params"]].split()[0]
    config["depth"] = lines[index["key_params"]].split()[1]
    config["G"] = lines[index["key_params"]].split()[2]
    config["nu"] = lines[index["key_params"]].split()[3]
    config["nu_u"] = lines[index["key_params"]].split()[4]
    config["B"] = lines[index["key_params"]].split()[5]
    config["D"] = lines[index["key_params"]].split()[6]

    return config, lines, index

def comsol_to_poel(comsol_values:dict):
    """
    由comsol中的参数推导POEL中的参数值（剪切模量、泊松比、不排水泊松比，skempton系数、水力扩散系数）的函数

    Parameters:\n
    comsol_values：一个字典，包含已知的comsol中的参数（杨氏模量、泊松比、流体可压缩性、孔隙率、流体动态黏度、渗透率、biot系数α）

    Returns:\n
    out：一个字典，包含推导所得的POEL程序的配置文件中对应于comsol的参数值所应该设置的参数值,键为参数名，值为参数值
    """

    E = comsol_values["E"]
    nu = comsol_values["nu"]
    chif = comsol_values["chif"]
    epsilon = comsol_values["epsilon"]
    eta = comsol_values["eta"]
    kappa = comsol_values["kappa"]
    alphaB = comsol_values["alphaB"]

    G = E / (2 * (1 + nu))
    K = E / (3 * (1 - 2 * nu))
    B = (1 / K - (1 - alphaB) / K) / (
        1 / K - (1 - alphaB) / K + epsilon * (chif - (1 - alphaB))
    )
    nu_u = (3 * nu + alphaB * B * (1 - 2 * nu)) / (3 - alphaB * B * (1 - 2 * nu))
    D = (2 * kappa * (1 - nu) * (1 + nu_u) * (1 + nu_u) * G * B * B) / (
        9 * eta * (1 - nu_u) * (nu_u - nu)
    )
    out = {"G": G, "nu": nu, "nu_u": nu_u, "B": B, "D": D}
    return out

def save_dictionary(path:str, dictionary:dict):
    """
    保存字典到文件
    
    参数:
    - path: 保存的文件路径
    - dictionary: 要保存的字典对象
    """
    with open(path+'.pkl', 'wb') as file:
        pickle.dump(dictionary, file)

def load_dictionary(path):
    """
    从文件中加载字典
    
    参数:
    - path: 文件路径
    
    返回值:
    - 加载的字典对象
    """
    with open(path, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def save_pkl(path:str,filename:str,data):
    """
    保存数据到二进制文件
    
    参数:
    - path: 保存的文件路径,形如“D:/code/”的字符串
    - filename: 保存的文件名,形如“mylist”的字符串
    - data: 要保存的数据
    """
    # 使用os.path.exists()检测目录是否存在
    if not os.path.exists(path):
        # 如果目录不存在，使用os.makedirs()递归创建目录
        os.makedirs(path)
    with open(path+f'{filename}.pkl', 'wb') as file:
        pickle.dump(data, file)

def load_pkl(path):
    """
    从文件中加载python数据对象
    
    参数:
    - path: 文件路径,包含文件名
    
    返回值:
    - data：从pkl文件中加载的python数据对象
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


# 读取注水、抽水、未抽出的水数据
def read_data(file_name):
    """
    读取注水、抽水、未抽出的水数据

    参数:
    - file_name: 文件路径,包含文件名

    返回值:
    - time：文件中读取的时间序列
    - water：文件中读取的抽注水数据
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
        lines = np.array([[float(val) for val in line.split()] for line in lines])
        time = lines[:, 0]
        water = lines[:, 1]

    return time, water

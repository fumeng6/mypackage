import numpy as np
from numpy import cos, sin
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp2d


def cylindrical_to_cartesian(theta: float | int, cy: np.ndarray):
    """
    将柱坐标系的对称张量变换至直角坐标系

    Parameters:\n
    theta：两坐标系之间R轴与X轴的夹角，单位为角度制

    cy：柱坐标系下的3×3张量矩阵

    Returns:\n
    ca：一个三维numpy矩阵,为转换至直角坐标系后的二阶张量
    """
    # 旋转角
    θ = np.array(theta)
    # 定义旋转矩阵
    R = np.array(
        [
            [cos(θ), sin(θ), 0],
            [sin(θ), -cos(θ), 0],
            [0, 0, -1],
        ]
    )
    # 将张量旋转到直角坐标系
    ca = R.T @ cy @ R
    return ca


def cartesian_to_cylindrical(theta: float | int, ca: np.ndarray):
    """
    直角坐标系对称张量转柱坐标系

    Parameters:\n
    theta：两坐标系之间R轴与X轴的夹角，单位为角度制

    ca：直角坐标系下的3×3张量矩阵

    Returns:\n
    cy：一个三维numpy矩阵,为转换至柱坐标系后的二阶张量
    """
    # 旋转角
    θ = np.array(theta)
    # 定义旋转矩阵
    R = np.array(
        [
            [cos(θ), sin(θ), 0],
            [sin(θ), -cos(θ), 0],
            [0, 0, -1],
        ]
    )
    # 将张量旋转到柱坐标系
    cy = R @ ca @ R.T
    return cy


def rotate_to_fault(strike: float | int, dip: float | int, tensor: np.ndarray):
    """
    将应力张量旋转到断层面上

    假设X轴指向正东，将X轴旋转（绕Z轴逆时针旋转）至断层走向上

    再绕新的X轴将Z轴旋转(逆时针)至垂直于断层面向上

    Parameters:\n
    strike：断层的走向，单位为角度制

    dip：断层的倾角，单位为角度制

    tensor：所需旋转的张量矩阵

    Returns:\n
    tensor_rotated：一个三维numpy矩阵,为旋转至断层面后的张量
    """
    # 定义断层的倾角和倾向角
    # 转成弧度制
    phi = np.radians(strike)
    delta = np.radians(dip)

    # 定义旋转矩阵
    # 假设X轴指向正东，将X轴旋转（绕Z轴逆时针旋转）至断层走向上
    R1 = np.array(
        [
            [sin(phi), -cos(phi), 0],
            [cos(phi), sin(phi), 0],
            [0, 0, 1],
        ]
    )

    # 再绕新的X轴将Z轴旋转(逆时针)至垂直于断层面向上
    R2 = np.array(
        [
            [1, 0, 0],
            [0, cos(delta), -sin(delta)],
            [0, sin(delta), cos(delta)],
        ]
    )
    # 总旋转矩阵
    R = R1 @ R2

    # 进行坐标变换
    # 注意应力张量是二阶张量，所以需要左乘和右乘旋转矩阵
    tensor_rotated = R @ tensor @ R.T

    return tensor_rotated


def combination_matrix(
    rr: float | int, tt: float | int, zz: float | int, zr: float | int
):
    """
    将柱坐标系下poel计算出的张量各分量组合成矩阵形式

    Parameters:\n
    rr：poel程序计算所得应变张量的rr分量

    tt：poel程序计算所得应变张量的rr分量

    zz：poel程序计算所得应变张量的zz分量

    zr：poel程序计算所得应变张量的zr分量

    Returns:\n
    matrix：一个三维numpy矩阵,由各分量组合而成的张量矩阵形式
    """
    rz = zr
    rt = tr = zt = tz = 0
    matrix = np.array([[rr, rt, rz], [tr, tt, tz], [zr, zt, zz]])
    return matrix


def load_poel_output(filename: str):
    """
    读取poel输出文件

    Parameters:\n
    filename：poel程序计算所得结果文件（包含文件名的完整路径）

    Returns:\n
    [data, time]：一个二元列表，0为poel程序计算所得结果，1为其计算的时间序列
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        del lines[0]
    lines = [x.split() for x in lines]
    lines = np.array(lines, dtype=float)
    data = lines[:, 1:]
    time = lines[:, 0:1]
    return [data, time]


def stress_tensor(strain_tensor: np.ndarray, E: float | int, nu: float | int):
    """
    由应变张量计算应力张量

    Parameters:\n
    strain_tensor：应变张量，是一个3×3的numpy矩阵

    E: 杨氏模量，以Pa为单位

    nu：泊松比

    Returns:\n
    stress_tensor：应力张量，是一个3×3的numpy矩阵
    """
    # 拉梅参数
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # 计算应变张量的迹
    trace_strain = np.trace(strain_tensor)

    # 计算应力张量
    stress_tensor = lam * trace_strain * np.eye(3) + 2 * mu * strain_tensor

    return stress_tensor


def degrees_to_dms(deg: float | int):
    """
    将角度值转换为度分秒值

    Parameters:\n
    deg：角度值

    Returns:\n
    dms：度分秒形式的角度值
    """
    d = int(deg)
    min_float = abs(deg - d) * 60
    min = int(min_float)
    sec = (min_float - min) * 60
    dms = f"{d}°{min}'{sec:.0f}\""

    return dms


def lon_lat_to_xy(
    lon: float | int, lat: float | int, lon0: float | int, lat0: float | int
):
    """
    经纬度坐标转距离坐标

    Parameters:\n
    lon：经度

    lat：纬度

    lon0：参考经度

    lat0：参考纬度

    Returns:\n
    [x,y]：x为沿纬度线上两点间的距离，y为沿经度线上两点间的距离
    """
    y = 111.199 * (lat - lat0)
    x = 111.199 * (lon - lon0) * cos(0.5 * (lat + lat0) * np.pi / 180.0)
    return [x, y]


def calculate_cfs(
    grid_x,
    grid_y,
    well_loc,
    pp,
    err,
    ett,
    ezz,
    ezr,
    E,
    nu,
    miu,
    phi,
    delta,
    lam,
):
    """
    输入二维网格面上的pp,err,ett,ezz,ezr网格点，在每个网格点上组合出应变张量，将各网格点的应变张量旋转至直角坐标系，\n
    通过应变张量及杨氏模量E和泊松比μ求出各网格点应力张量，将各网格点应力张量旋转至断层坐标系，随后计算正剪应力，通过正剪应力及孔隙压力计算出二维网格面上各点的库伦应力。

        Parameters:\n
        - grid_x: 二维网格面的X坐标数组。
        - grid_y: 二维网格面的Y坐标数组。
        - well_loc: 注水井在网格面上的坐标，初始应该为[0,0]。
        - pp: 孔隙压力。
        - err: 应变rr分量。
        - ett: 应变tt分量。
        - ezz: 应变zz分量。
        - ezr: 应变zr分量。
        - E: 杨氏模量（Pa）。
        - nu: 泊松比。
        - miu: 断层摩擦系数。
        - phi: 断层走向（°）。
        - delta: 断层倾角（°）。
        - lam: 断层滑动角（°）。

        Returns:\n
        - cfs: 一个二维网格，其中各点的值为该点的库伦应力
        - elastic_stress_part：一个二维网格，其中各点的值为该点的库伦应力中的弹性应力部分
        - pore_pressure_part：一个二维网格，其中各点的值为该点的库伦应力中的孔隙压力部分
        - tau_part: 一个二维网格，其中各点的值为该点的库伦应力中的剪切应力部分
        - sigma_part: 一个二维网格，其中各点的值为该点的库伦应力中的正应力部分
    """
    # 设置待用的空数组容器
    # 储存应变张量的数组
    strain_collection_matrix = np.empty(pp.shape, dtype=object)
    # 储存直角坐标系下应变张量的数组
    strain_ca_collection_matrix = np.empty(pp.shape, dtype=object)
    # 储存直角坐标系下应力张量的数组
    stress_ca_collection_matrix = np.empty(pp.shape, dtype=object)
    # 储存断层坐标系下应力张量的数组
    stress_duan_collection_matrix = np.empty(pp.shape, dtype=object)
    # 储存正应力的数组
    normal_stress = np.zeros(pp.shape)
    # 储存剪切应力的数组
    shear_stress = np.zeros(pp.shape)

    # 循环遍历网格上每一点，计算其断层坐标系下的应力张量
    for i in range(pp.shape[0]):
        for j in range(pp.shape[1]):
            # 将poel计算出的应变各分量组合为张量
            strain_collection_matrix[i, j] = combination_matrix(
                err[i, j], ett[i, j], ezz[i, j], -ezr[i, j]
            )

            # 将poel中的柱坐标系下的应变张量变换至直角坐标系下
            # 确定这一点相对于poel中注水点的极角
            theta = np.arctan2(grid_y[i, j] - well_loc[1], grid_x[i, j] - well_loc[0])
            # 根据极角进行坐标系变换
            strain_ca_collection_matrix[i, j] = cylindrical_to_cartesian(
                theta, strain_collection_matrix[i, j]
            )

            # 计算直角坐标系下应力张量
            stress_ca_collection_matrix[i, j] = stress_tensor(
                strain_ca_collection_matrix[i, j], E, nu
            )

            # 将应力张量旋转至断层坐标系
            stress_duan_collection_matrix[i, j] = rotate_to_fault(
                phi, delta, stress_ca_collection_matrix[i, j]
            )

            # 计算正剪应力
            # 将滑动角转换为弧度制
            rake = np.deg2rad(lam)
            normal_stress[i, j] = stress_duan_collection_matrix[i, j][2, 2]
            shear_stress[i, j] = stress_duan_collection_matrix[i, j][2, 0] * np.cos(
                rake
            ) + stress_duan_collection_matrix[i, j][2, 1] * np.sin(rake)

    # 计算库伦应力
    cfs = shear_stress + miu * (normal_stress + pp)
    elastic_stress_part = shear_stress + miu * normal_stress
    pore_pressure_part = miu * pp
    tau_part = shear_stress
    sigma_part = miu *normal_stress
    return cfs, elastic_stress_part, pore_pressure_part,tau_part,sigma_part


def generate_interpolated_grid(
    radius: int | float,
    density: int,
    well_location: list,
    data_for_interpolation: list | np.ndarray,
    xy: bool,
):
    """
    给定网格的半径和密度、注入井的位置和用于插值的数据，即可获得三次样条插值之后网格面，

        Parameters:\n
        - radius: 圆形网格的半径。
        - density: 沿半径的网格点数。
        - well_location: 注水井的（x，y）坐标。
        - data_for_interpolation: 沿半径进行插值的每个原始网格点的数据值。
        - xy：是否输出X,Y坐标网格。

        Returns:\n
        - X: 笛卡尔坐标系中网格点的 x 坐标。
        - Y: 笛卡尔坐标系中网格点的 y 坐标。
        - grid_values: 每个网格点的插值结果,是一个尺寸为（density×density）的numpy数组。
    """

    # 生成网格的极坐标
    theta = np.linspace(0, 2 * np.pi, density)
    r = np.linspace(0, radius, density)

    # 创建极坐标网格
    T, R = np.meshgrid(theta, r)

    # 将极坐标转换为笛卡尔坐标
    X, Y = R * np.cos(T), R * np.sin(T)

    # 计算每个网格点到注入井的距离
    grid_rr = np.sqrt((X - well_location[0]) ** 2 + (Y - well_location[1]) ** 2)

    # 定义要插值的数据的原始半径坐标
    r_coordinates = np.linspace(0, radius, len(data_for_interpolation))

    # 使用 CubicSpline 执行三次样条插值
    cs = CubicSpline(r_coordinates, data_for_interpolation, extrapolate=True)
    grid_values = cs(grid_rr)
    if xy:
        return X, Y, grid_values
    else:
        return grid_values


def inter2d_zr(
    len_z, len_r, samples_z, samples_r, new_samples_z, new_samples_r, data_zr
):
    """
    给定ZR面的原始尺寸、Z向及R向的目标网格密度、用于插值的数据，即可获得三次样条插值之后ZR网格面，

        Parameters:\n
        - len_z: ZR网格面Z向的长度，单位为千米。
        - len_r: ZR网格面R向的长度，单位为千米。
        - samples_z: 原始数据沿Z向的网格点数。
        - samples_r: 原始数据沿R向的网格点数。
        - new_samples_z: 目标网格面沿Z向的网格点数。
        - new_samples_r: 目标网格面沿R向的网格点数。
        - data_zr: 原始ZR网格面数据。

        Returns:\n
        - new_zr_values: 每个网格点的插值结果,是一个尺寸为(new_samples_z, new_samples_r)的numpy数组。
    """
    # 对原始ZR面网格进行插值获得更密的ZR面数据
    # 创建原始的网格坐标
    z_coords = np.linspace(0, len_z, samples_z)
    r_coords = np.linspace(0, len_r, samples_r)

    # 定义新的更密网格的坐标
    new_z_coords = np.linspace(0, len_z, new_samples_z)
    new_r_coords = np.linspace(0, len_r, new_samples_r)

    # 创建插值函数
    interpolator = interp2d(r_coords, z_coords, data_zr, kind="cubic")

    # 在新的网格上计算插值后的值
    new_zr_values = interpolator(new_r_coords, new_z_coords)

    return new_zr_values

# 生成平面插值网格

def r_plane_interpolation(radius,samples,grid_x, grid_y, well_location, data_for_interpolation):
    """
    给定网格的范围和密度、注入井的位置和用于插值的数据，即可获得三次样条插值之后网格面R_mesh。

        Parameters:\n
        - radius: 原始径向数据的最大范围
        - samples: 原始径向数据的采样数
        - grid_x: 插值平面的X网格，二维。
        - grid_y: 插值平面的Y网格，二维。
        - well_location: 注水井的（x，y）坐标。
        - data_for_interpolation: 待插值的每个原始网格点的数据值。

        Returns:\n
        - grid_r: 插值面每个网格点距离注水井的距离网格，二维。
        - grid_result: 每个网格点的插值结果,是一个尺寸为与grid_x,grid_y相同的numpy数组。
    """

    # 计算每个网格点到注入井的距离
    well_x = round(well_location[0], 1)
    well_y = round(well_location[1], 1)
    grid_r = np.sqrt((grid_x - well_x) ** 2 + (grid_y - well_y) ** 2)

    r_coordinates = np.linspace(0, radius, samples)
    grid_result = np.interp(grid_r, r_coordinates, data_for_interpolation)
    # # 定义要插值的数据的原始半径坐标
    # r_coordinates = np.linspace(0, np.max(grid_r), len(data_for_interpolation))

    # # 使用 CubicSpline 执行三次样条插值
    # cs = CubicSpline(r_coordinates, data_for_interpolation, extrapolate=True)
    # grid_values = cs(grid_r)

    return grid_r, grid_result

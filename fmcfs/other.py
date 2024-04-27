import math
from datetime import datetime, timedelta
import numpy as np
import re
import os

def scale_change(value: float | int, target_magnitude: int):
    """
    将输入的值从一个数量级放缩到另一个指定数量级

    Parameters:\n
    value：原始值

    target_magnitude：目标数量级，即10的n次方

    Returns:\n
    [scaled_value,scale_diff]：前者为放缩过后的值，后者为放缩前后的数量级差值
    """
    current_magnitude = math.floor(math.log10(abs(value)))
    scale_diff = target_magnitude - current_magnitude
    scale_factor = 10**scale_diff
    scaled_value = value * scale_factor

    return [scaled_value, scale_diff]


def time2date(target_time: float):
    """
    将一个形如2006.837的时间转换成年月日时分秒的形式

    Parameters:\n
    target_time：需要转换的时间

    Returns:\n
    final_date：完成转换的目标时间，形如年月日时分秒
    """
    year = int(np.floor(target_time))

    less_than_one_year = target_time - year
    day_of_year = less_than_one_year * 365.25
    day = int(np.floor(day_of_year))

    less_than_one_day = day_of_year - day

    # 计算日期
    date = datetime(year, 1, 1) + timedelta(days=day - 1)
    time = timedelta(days=less_than_one_day)
    final_date = date + time
    # 格式化一下
    final_date = final_date.strftime("%Y-%m-%d %H:%M:%S")

    return final_date


def date2time(datetime_str):
    """
    将日期时间字符串转换为小数形式的年份
    例如："2006-11-01 17:08:31" 转换为 2006.837

    参数：
    datetime_str: 日期时间字符串，格式为 "YYYY-MM-DD HH:MM:SS"

    返回值：
    decimal_year: 小数形式的年份
    """
    # 将字符串解析为 datetime 对象
    datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

    # 计算年份和天数的小数部分
    year = datetime_obj.year
    day_of_year = datetime_obj.timetuple().tm_yday
    hour = datetime_obj.hour
    minute = datetime_obj.minute

    # 计算小时和分钟的小数部分
    hour_fraction = hour + minute / 60.0

    # 计算结果
    decimal_year = year + (day_of_year - 1 + hour_fraction / 24.0) / 365.0

    return decimal_year


def get_event_xy(catalog_path: str, key_list: list, lon_range: list, lat_range: list):
    """
    将地震事件的经纬度转换为地理坐标

    Parameters:\n
    catalog_path：需要转换的地震事件目录的路径

    key_list：作为返回字典的键名的列表

    Returns:\n
    event：完成转换的地震事件字典，其中包含x,y坐标值，
    """
    with open(catalog_path, "r") as f:
        lines = f.readlines()

    event_x = []
    event_y = []
    event = {key: {"x": [], "y": []} for key in key_list}
    event[">=5"] = {"x": [], "y": []}
    for i in range(len(lines)):
        # 拆分每行数据的字段
        line = lines[i].split()
        # 解析字段并转换为适当的数据类型
        longitude = float(line[0])
        latitude = float(line[1])
        depth = float(line[4])
        y = 111.199 * (latitude - lat_range[0])
        x = (
            111.199
            * (longitude - lon_range[0])
            * np.cos(0.5 * (latitude + lat_range[0]) * np.pi / 180.0)
        )
        event_x.append(x * 10)
        event_y.append(y * 10)
        if depth > 0 and depth < 2:
            event["1km"]["x"].append(x * 10)
            event["1km"]["y"].append(y * 10)
        elif depth > 1 and depth < 3:
            event["2km"]["x"].append(x * 10)
            event["2km"]["y"].append(y * 10)
        elif depth > 2 and depth < 4:
            event["3km"]["x"].append(x * 10)
            event["3km"]["y"].append(y * 10)
        elif depth > 3 and depth < 5:
            event["4km"]["x"].append(x * 10)
            event["4km"]["y"].append(y * 10)
        elif depth >= 5:
            event[">=5"]["x"].append(x * 10)
            event[">=5"]["y"].append(y * 10)

    return event


def get_event_xy2(
    catalog_path: str, depth_control: int | float, lon_range: list, lat_range: list
):
    """
    将地震事件的经纬度转换为地理坐标

    Parameters:\n
    catalog_path：需要转换的地震事件目录的路径

    key_list：作为返回字典的键名的列表

    Returns:\n
    event：完成转换的地震事件字典，其中包含x,y坐标值列表，
    """
    with open(catalog_path, "r") as f:
        lines = f.readlines()

    event_x = []
    event_y = []
    event = {"x": [], "y": []}

    # 使用正则表达式来提取数字部分
    match = re.search(r"\d+", depth_control)
    deep = int(match.group())

    for i in range(len(lines)):
        # 拆分每行数据的字段
        line = lines[i].split()
        # 解析字段并转换为适当的数据类型
        longitude = float(line[0])
        latitude = float(line[1])
        depth = float(line[4])
        y = 111.199 * (latitude - lat_range[0])
        x = (
            111.199
            * (longitude - lon_range[0])
            * np.cos(0.5 * (latitude + lat_range[0]) * np.pi / 180.0)
        )
        event_x.append(x * 10)
        event_y.append(y * 10)
        if depth > deep - 1 and depth < deep + 1:
            event["x"].append(x * 10)
            event["y"].append(y * 10)

    return event


def degrees_to_km_by_haversine(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度坐标之间的大圆距离（千米）

    参数：

    lon1, lat1: 第一个点的经纬度坐标

    lon2, lat2: 第二个点的经纬度坐标

    返回值：

    distance:两经纬度坐标点之间的大圆距离（千米）
    """
    # 将经纬度转换为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine 公式计算距离
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    radius = 6371  # 地球半径（千米）
    distance = radius * c
    return distance


def recursive_print_dict(d: dict, indent=""):
    """
    递归打印嵌套字典的键名

    递归遍历字典中的所有键名，包括嵌套字典的键名，并在每个键名前添加指定的缩进。

    参数：
    d (dict): 要遍历的字典。
    indent (str, 可选): 用于表示每个层级的缩进字符串。默认为空字符串。

    返回值：
    无。函数用于打印键名，而不返回任何值。
    """
    for key, value in d.items():
        key_str = str(key)
        print(indent + key_str)
        if isinstance(value, dict):
            recursive_print_dict(value, indent + "-")


def is_leap_year(year: int | float):
    """
    判断输入年份是否为闰年

    参数：
    year (int|float): 要判断的年份。

    返回值：
    布尔值：是闰年返回True，否则返回False。
    """
    year = int(year)
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False


def injection_data_processing(lines: list, base_time: int | float):
    """
    对抽水和注水数据进行处理，以符合poel中的国际单位制

    参数：
    lines: 要处理的抽/注水数据列表，每个子列表为一行，子列表第一个元素为时间（年），第二个元素为抽/注水速率（m³/月）。

    base_time: 基准时间，单位为秒,处理后抽/注水数据皆从此开始。

    返回值：
    new_lines：处理完成的数组形式的抽/注水数据。
    """
    # 设置一年的总秒数
    seconds = 365.25 * 24 * 60 * 60
    # 设置闰年每月天数
    leapdays = 366 / 12
    # 设置非闰年每月天数
    no_leapdays = 365 / 12
    # 注水数据处理
    new_lines = []
    for i, line in enumerate(lines):
        # 计算与基准时间的差值
        temp_time = round((line[0] - base_time) * seconds)
        # 获取本条数据对应年份每月天数
        days = leapdays if is_leap_year(int(line[0])) else no_leapdays
        # 计算每秒的水量
        temp_water = round(line[1] * 1e4 / (days * 24 * 60 * 60), 5)
        # 将转换结果添加入新的列表中
        new_lines.append([temp_time, temp_water])
    new_lines = np.array(new_lines)
    return new_lines


def filter_events_within_radius(catalog, center, radius_km):
    """
    筛选出距离中心点一定半径内的地震事件

    参数：\n
    catalog: 地震目录数组，假设经纬度分别在第0和第1列\n
    center: 中心点的经纬度坐标（经度, 纬度）\n
    radius_km: 半径（千米）\n

    返回值：\n
    filtered_catalog：按区域范围筛选后的地震目录
    """
    filtered_catalog = []

    for event in catalog:
        event_lon, event_lat = event[0], event[1]
        distance = degrees_to_km_by_haversine(
            center[0], center[1], event_lon, event_lat
        )
        if distance <= radius_km:
            filtered_catalog.append(event)

    filtered_catalog = np.array(filtered_catalog)
    return filtered_catalog


def sum_quakes_per_month(year_start, year_end, catalog):
    """
    从地震目录中归纳出指定年限内每月地震数

    参数：\n
    year_start：开始年份
    year_end：结束年份（例如想要筛查的是2000到2012年末，那么开始就是2000，结束就是2012）
    catalog: 地震目录数组，须符合年月日在数组的末尾三列\n

    返回值：\n
    x_positions：绘图时所用的X轴坐标，代表该年该月
    earthquakes：筛查完毕的每月的地震事件数量
    """

    year_list = [x for x in range(year_start, year_end + 1)]
    month_list = [x for x in range(1, 13)]
    event_year = catalog[:, -3]

    # 设置储存每月地震频数的容器
    data = []
    for year in year_list:
        # 条件筛选
        mask_year = event_year == year
        temp = catalog[mask_year]
        event_month = temp[:, -2]
        for month in month_list:
            mask_month = event_month == month
            temp_month = event_month[mask_month]
            data.append((year, month, len(temp_month)))

    # 将输入数据分割成年份和月份列表
    years_months = [f"{item[0]}-{item[1]}" for item in data]
    earthquakes = [item[2] for item in data]
    # 计算每个柱的 x 坐标位置
    x_positions = np.arange(len(years_months)) / 12 + year_start

    return x_positions, earthquakes


def get_next_filename(directory, prefix="s", extension=".png"):
    """
    获取给定目录下指定前缀和扩展名的文件的下一个序号的文件名。

    参数:
    - directory: 要搜索的目录路径。
    - prefix: 文件名的前缀。
    - extension: 文件的扩展名。

    返回:
    - 新文件的完整路径，其序号为当前最大序号加一。
    """
    # 列出给定目录下的所有文件
    files = os.listdir(directory)

    # 过滤出以指定前缀开头并以指定扩展名结尾的文件，并去掉前缀和扩展名获取序号
    numbers = []
    for filename in files:
        if filename.startswith(prefix) and filename.endswith(extension):
            number_part = filename[len(prefix) : -len(extension)]
            if number_part.isdigit():
                numbers.append(int(number_part))

    # 如果目录中没有符合条件的文件，就从1开始
    if not numbers:
        next_number = 1
    else:
        # 找出最大的序号并加一
        next_number = max(numbers) + 1

    # 生成新的文件名
    new_filename = f"{prefix}{next_number}{extension}"
    return os.path.join(directory, new_filename)

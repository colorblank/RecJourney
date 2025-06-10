import math
from datetime import datetime, timezone

from xxhash import xxh32


def str_hash(x: str, num_embeddings: int, seed: int | None = None) -> int:
    """
    对字符串进行哈希。

    Args:
        x: 输入字符串。
        num_embeddings: 嵌入空间的大小。
        seed: 哈希种子。

    Returns:
        哈希后的整数值。
    """
    seed = 1024 if seed is None else seed
    return xxh32(str(x), seed).intdigest() % num_embeddings


def str_to_list(x: str, sep: str) -> list[str]:
    """
    将字符串按分隔符转换为字符串列表。

    Args:
        x: 输入字符串。
        sep: 分隔符。

    Returns:
        字符串列表。
    """
    return str.split(x, sep=sep)


def list_str_split(x: list[str], sep: str) -> list[list[str]]:
    """
    将字符串列表中的每个字符串按分隔符转换为字符串列表的列表。

    Args:
        x: 输入字符串列表。
        sep: 分隔符。

    Returns:
        字符串列表的列表。
    """
    res = [s.split(sep) for s in x]
    return res


def list_hash(x: list[str], num_embeddings: int, seed: int | None = None) -> list[int]:
    """
    对字符串列表中的每个字符串进行哈希。

    Args:
        x: 输入字符串列表。
        num_embeddings: 嵌入空间的大小。
        seed: 哈希种子。

    Returns:
        哈希后的整数列表。
    """
    seed = 1024 if seed is None else seed
    return [str_hash(i, num_embeddings, seed) for i in x]


def padding(x: list[str], max_len: int, padding_value: str) -> list[str]:
    """
    对列表进行填充。

    Args:
        x: 输入列表。
        max_len: 填充后的最大长度。
        padding_value: 填充值。

    Returns:
        填充后的列表。
    """
    return x[:max_len] + [padding_value] * (max_len - len(x))


def log1p(x: int | float) -> float:
    """
    计算 log(1 + x)。

    Args:
        x: 输入数值。

    Returns:
        计算结果。
    """
    return math.log1p(float(x))


def floor_divde(x: int, divide: int) -> int:
    """
    计算整数除法。

    Args:
        x: 被除数。
        divide: 除数。

    Returns:
        整数除法结果。
    """
    return x // divide


def clip(x: int | float, min_value: int | float, max_value: int | float) -> int | float:
    """
    将数值裁剪到指定范围。

    Args:
        x: 输入数值。
        min_value: 最小值。
        max_value: 最大值。

    Returns:
        裁剪后的数值。
    """
    return min(max_value, max(min_value, x))


def int_to_date(x: int, tz=None) -> datetime:
    """
    将整数时间戳转换为 datetime 对象。

    Args:
        x: 整数时间戳。
        tz: 时区。

    Returns:
        datetime 对象。
    """
    time_zone = timezone.utc if tz is None else tz
    return datetime.fromtimestamp(x, tz=time_zone)


def str_to_date(x: str, format: str) -> datetime:
    """
    将字符串日期转换为 datetime 对象。

    Args:
        x: 字符串日期。
        format: 日期格式。

    Returns:
        datetime 对象。
    """
    try:
        return datetime.strptime(x, format)
    except ValueError:
        # 如果原始格式不匹配，尝试另一种常见格式
        if format == "%Y%m%d":
            return datetime.strptime(x, "%Y-%m-%d")
        elif format == "%Y-%m-%d":
            return datetime.strptime(x, "%Y%m%d")
        else:
            raise  # 如果都不是，则重新抛出原始错误


def get_day(x: datetime) -> int:
    """
    获取日期中的天。

    Args:
        x: datetime 对象。

    Returns:
        天数。
    """
    return x.day


def get_hour(x: datetime) -> int:
    """
    获取日期中的小时。

    Args:
        x: datetime 对象。

    Returns:
        小时数。
    """
    return x.hour


def get_minute(x: datetime) -> int:
    """
    获取日期中的分钟。

    Args:
        x: datetime 对象。

    Returns:
        分钟数。
    """
    return x.minute


def get_month(x: datetime) -> int:
    """
    获取日期中的月份。

    Args:
        x: datetime 对象。

    Returns:
        月份数。
    """
    return x.month


def isoweekday(x: datetime) -> int:
    """
    获取日期中的周几（ISO 标准）。

    Args:
        x: datetime 对象。

    Returns:
        周几（1=周一，7=周日）。
    """
    return x.isoweekday()



import math
from datetime import datetime, timezone
from typing import List, Optional, Union

from xxhash import xxh32


def str_hash(x: str, num_embeddings: int, seed: Optional[int] = None) -> int:
    seed = 1024 if seed is None else seed
    return xxh32(str(x), seed).intdigest() % num_embeddings


def str_to_list(x: str, sep: str) -> List[int]:
    return str.split(x, sep=sep)


def list_str_split(x: List[str], sep: str) -> List[List[str]]:
    res = [s.split(sep) for s in x]
    return res


def list_hash(
    x: List[str], num_embeddings: int, seed: Optional[int] = None
) -> List[int]:
    seed = 1024 if seed is None else seed
    return [str_hash(i, num_embeddings, seed) for i in x]


def padding(x: List[str], max_len: int, padding_value: str) -> List[str]:
    return x[:max_len] + [padding_value] * (max_len - len(x))


def log1p(x: Union[int, float]) -> float:
    return math.log1p(float(x))


def floor_divde(x: int, divide: int) -> int:
    return x // divide


def clip(
    x: Union[int, float], min_value: Union[int, float], max_value: Union[int, float]
) -> Union[int, float]:
    return min(max_value, max(min_value, x))


def int_to_date(x: int, tz=None) -> datetime:
    time_zone = timezone.utc if tz is None else tz
    return datetime.fromtimestamp(x, tz=time_zone)


def str_to_date(x: str, format: str) -> datetime:
    return datetime.strptime(x, format)


def get_day(x: datetime) -> int:
    """一个月的第几天

    Args:
        x (datetime): _description_

    Returns:
        int: _description_
    """
    return x.day


def get_hour(x: datetime) -> int:
    """当天的小时

    Args:
        x (datetime): _description_

    Returns:
        int: _description_
    """
    return x.hour


def get_minute(x: datetime) -> int:
    """当前小时的第多少分钟

    Args:
        x (datetime): _description_

    Returns:
        int: _description_
    """
    return x.min


def get_month(x: datetime) -> int:
    """第几个月

    Args:
        x (datetime): _description_

    Returns:
        int: _description_
    """
    return x.month


def isoweekday(x: datetime) -> int:
    """周几

    Args:
        x (datetime): _description_

    Returns:
        int: _description_
    """
    return x.isoweekday()

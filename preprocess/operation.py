import math
from typing import Iterable, List, Union

from xxhash import xxh32


def str_hash(x: str, num_embeddings: int, seed: int = 1024) -> int:
    assert not isinstance(x, Iterable), f"input {x} can not be Iterable."
    return xxh32(str(x), seed).intdigest() % num_embeddings


def str_to_list(x: str, sep: str) -> List[int]:
    return str.split(x, sep=sep)


def list_hash(x: List[str], num_embeddings: int, seed: int = 1024) -> List[int]:
    return [str_hash(i, num_embeddings, seed) for i in x]


def padding(x: List[str], max_len: int, padding_value: str) -> List[str]:
    return x[:max_len] + [padding_value] * (max_len - len(x))


def log1x(x: Union[int, float], base: float = math.e) -> float:
    return math.log(float(x) + 1, base=base)


def floor_divde(x: int, divide: int) -> int:
    return x // divide


def clip(
    x: Union[int, float], min_value: Union[int, float], max_value: Union[int, float]
) -> Union[int, float]:
    return min(max_value, max(min_value, x))

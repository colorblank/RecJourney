from xxhash import xxh32
from typing import List, Iterable


def str_hash(x: str, num_embeddings: int, seed: int = 1024):
    assert not isinstance(x, Iterable), f"input {x} can not be Iterable."
    return xxh32(str(x), seed).intdigest() % num_embeddings


def str_to_list(x: str, sep: str) -> List[int]:
    return str.split(x, sep=sep)

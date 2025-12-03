from functools import singledispatch
from typing import Any


@singledispatch
def _serialize_for_json(obj: Any) -> Any:
    return obj


@_serialize_for_json.register
def _(obj: dict) -> dict:
    return {k: _serialize_for_json(v) for k, v in obj.items()}


@_serialize_for_json.register
def _(obj: list) -> list:
    return [_serialize_for_json(item) for item in obj]


import struct


def _to_float32_bytes(vec) -> bytes:
    return b"".join(struct.pack("<f", x) for x in vec)

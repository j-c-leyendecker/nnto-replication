"""Minimal, dependency-free reader for TensorFlow "tensor bundle" checkpoints.

The original repository ships its trained Keras models in the TF SavedModel
format (``variables/variables.index`` + ``variables.data-00000-of-00001``).
TensorFlow is a heavy dependency (and is not required for anything else in
this package), so this module parses the two files directly:

* ``variables.index`` is a LevelDB-style SSTable whose values are serialized
  ``BundleEntryProto`` messages (dtype, shape, shard, offset, size, crc32c).
* ``variables.data-*`` holds the raw little-endian tensor bytes.

Only what is needed to read dense numeric tensors is implemented.  Every
tensor's CRC32C (masked, as TensorFlow stores it) is verified so that a
mis-parse can never silently return wrong weights.
"""
from __future__ import annotations

import os
import struct
from typing import Dict, Tuple

import numpy as np

_TABLE_MAGIC = 0xDB4775248B80FB57

# tensorflow/core/framework/types.proto
_DTYPES = {
    1: np.dtype("<f4"),
    2: np.dtype("<f8"),
    3: np.dtype("<i4"),
    4: np.dtype("u1"),
    5: np.dtype("<i2"),
    6: np.dtype("i1"),
    9: np.dtype("<i8"),
    10: np.dtype("?"),
    17: np.dtype("<u2"),
    22: np.dtype("<u4"),
    23: np.dtype("<u8"),
}
_DT_STRING = 7


# --------------------------------------------------------------------------- #
# CRC32C (Castagnoli), table driven; TensorFlow stores a *masked* value.
# --------------------------------------------------------------------------- #
def _make_crc_table():
    poly = 0x82F63B78
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ poly if c & 1 else c >> 1
        table.append(c)
    return table


_CRC_TABLE = _make_crc_table()


def crc32c(data: bytes) -> int:
    c = 0xFFFFFFFF
    for b in data:
        c = _CRC_TABLE[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    c = crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


# --------------------------------------------------------------------------- #
# varint / protobuf helpers
# --------------------------------------------------------------------------- #
def _varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError("malformed varint")


def _parse_proto(buf: bytes) -> Dict[int, list]:
    """Parse one protobuf message into {field_number: [raw values]}."""
    out: Dict[int, list] = {}
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = _varint(buf, pos)
        elif wire == 1:
            val = struct.unpack_from("<Q", buf, pos)[0]
            pos += 8
        elif wire == 2:
            ln, pos = _varint(buf, pos)
            val = buf[pos:pos + ln]
            pos += ln
        elif wire == 5:
            val = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out.setdefault(field, []).append(val)
    return out


def _signed64(v: int) -> int:
    return v - (1 << 64) if v >= (1 << 63) else v


def _parse_shape(buf: bytes) -> Tuple[int, ...]:
    msg = _parse_proto(buf)
    dims = []
    for d in msg.get(2, []):
        dm = _parse_proto(d)
        dims.append(_signed64(dm.get(1, [0])[0]))
    return tuple(dims)


# --------------------------------------------------------------------------- #
# SSTable reading
# --------------------------------------------------------------------------- #
def _read_block(buf: bytes, offset: int, size: int) -> bytes:
    block = buf[offset:offset + size]
    ctype = buf[offset + size]
    if ctype != 0:
        raise NotImplementedError(
            "compressed SSTable blocks are not supported (compression type "
            f"{ctype}); re-save the checkpoint uncompressed")
    stored = struct.unpack_from("<I", buf, offset + size + 1)[0]
    if masked_crc32c(block + bytes([ctype])) != stored:
        raise ValueError("SSTable block checksum mismatch")
    return block


def _iter_block(block: bytes):
    num_restarts = struct.unpack_from("<I", block, len(block) - 4)[0]
    end = len(block) - 4 - 4 * num_restarts
    pos = 0
    key = b""
    while pos < end:
        shared, pos = _varint(block, pos)
        non_shared, pos = _varint(block, pos)
        vlen, pos = _varint(block, pos)
        key = key[:shared] + block[pos:pos + non_shared]
        pos += non_shared
        value = block[pos:pos + vlen]
        pos += vlen
        yield key, value


def read_index(index_path: str) -> Dict[str, dict]:
    """Return {tensor_name: {dtype, shape, shard, offset, size, crc}}."""
    with open(index_path, "rb") as fh:
        buf = fh.read()
    if len(buf) < 48:
        raise ValueError("file too small to be an SSTable")
    footer = buf[-48:]
    magic = struct.unpack_from("<Q", footer, 40)[0]
    if magic != _TABLE_MAGIC:
        raise ValueError("not a TensorFlow checkpoint index (bad magic)")
    pos = 0
    _, pos = _varint(footer, pos)          # metaindex offset
    _, pos = _varint(footer, pos)          # metaindex size
    idx_off, pos = _varint(footer, pos)
    idx_size, pos = _varint(footer, pos)

    entries: Dict[str, dict] = {}
    for _, handle in _iter_block(_read_block(buf, idx_off, idx_size)):
        b_off, p = _varint(handle, 0)
        b_size, p = _varint(handle, p)
        for key, value in _iter_block(_read_block(buf, b_off, b_size)):
            if key == b"":                 # BundleHeaderProto
                hdr = _parse_proto(value)
                if hdr.get(2, [0])[0] != 0:
                    raise NotImplementedError("big-endian bundles unsupported")
                continue
            msg = _parse_proto(value)
            entries[key.decode("utf-8")] = {
                "dtype": msg.get(1, [0])[0],
                "shape": _parse_shape(msg[2][0]) if 2 in msg else (),
                "shard": msg.get(3, [0])[0],
                "offset": msg.get(4, [0])[0],
                "size": msg.get(5, [0])[0],
                "crc": msg.get(6, [None])[0],
            }
    return entries


def load_variables(variables_dir: str, verify_crc: bool = True) -> Dict[str, np.ndarray]:
    """Load every numeric tensor of a SavedModel ``variables`` directory."""
    index_path = os.path.join(variables_dir, "variables.index")
    entries = read_index(index_path)
    shards: Dict[int, bytes] = {}
    out: Dict[str, np.ndarray] = {}
    for name, e in entries.items():
        if e["dtype"] == _DT_STRING or e["dtype"] not in _DTYPES:
            continue
        if e["shard"] not in shards:
            cands = [f for f in os.listdir(variables_dir)
                     if f.startswith(f"variables.data-{e['shard']:05d}-of-")]
            if len(cands) != 1:
                raise FileNotFoundError(f"data shard {e['shard']} not found")
            with open(os.path.join(variables_dir, cands[0]), "rb") as fh:
                shards[e["shard"]] = fh.read()
        raw = shards[e["shard"]][e["offset"]:e["offset"] + e["size"]]
        if len(raw) != e["size"]:
            raise ValueError(f"truncated tensor data for {name}")
        if verify_crc and e["crc"] is not None and masked_crc32c(raw) != e["crc"]:
            raise ValueError(f"CRC mismatch for tensor {name}")
        dt = _DTYPES[e["dtype"]]
        arr = np.frombuffer(raw, dtype=dt)
        n_expected = int(np.prod(e["shape"])) if e["shape"] else 1
        if arr.size != n_expected:
            raise ValueError(f"size mismatch for tensor {name}")
        out[name] = arr.reshape(e["shape"]).copy()
    return out

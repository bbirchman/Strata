"""P40 page protocol v1: length-prefixed blob transport over a Unix socket.

No GPU, no network, no model files: the server synthesizes deterministic
fixture blobs (same geometry as the engine: 1,382,400 bytes per expert)
so the client can verify byte-exactness locally.

Framing (little-endian, borrowed from the stage_link/two-pc shape):
    [u32 magic 0x50343054 "P40T"][u32 op][u64 len][payload]
Payloads larger than MAX_PAYLOAD are refused before allocation.

Ops:
    HELLO=1       {u32 version, u32 namelen, name} -> HELLO_REPLY
    HELLO_REPLY=2 {u32 version, u64 blob_bytes, u64 layers, u64 experts}
    GET=3         {u64 layer, u64 expert} -> BLOB or MISS
    BLOB=4        {u64 layer, u64 expert, 32B sha256, blob bytes}
    MISS=5        {u64 layer, u64 expert}
    PREFETCH=6    {u32 n, n x (u64 layer, u64 expert)} -> ACK
    ACK=7         {u64 code}
    STATS=8       {} -> STATS_REPLY
    STATS_REPLY=9 {u64 gets, u64 hits, u64 misses, u64 bytes_served}
    BYE=10        {} (server closes; no reply)
    ERROR=11      {u32 msglen, msg}
"""
from __future__ import annotations

import hashlib
import os
import socket
import struct
import threading
from socketserver import BaseRequestHandler, ThreadingUnixStreamServer

MAGIC = 0x50343054
VERSION = 1
# Engine geometry, include/strata/core/expert_source.hpp: H = 2560, FF = 640.
BLOB_BYTES = 1382400
LAYERS = 48
EXPERTS = 512
MAX_PAYLOAD = 8 << 20

HELLO, HELLO_REPLY, GET, BLOB, MISS, PREFETCH, ACK, STATS, STATS_REPLY, BYE, ERROR = range(1, 12)

_HEADER = struct.Struct("<IIQ")


def encode_frame(op: int, payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload {len(payload)} exceeds {MAX_PAYLOAD}")
    return _HEADER.pack(MAGIC, op, len(payload)) + payload


def _recvn(f, n: int) -> bytes:
    chunks = []
    while n > 0:
        chunk = f.read(n)
        if not chunk:
            raise ConnectionError("peer closed mid-frame")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def decode_frame(f):
    """Read one frame from a buffered reader. Returns (op, payload)."""
    header = _recvn(f, _HEADER.size)
    magic, op, length = _HEADER.unpack(header)
    if magic != MAGIC:
        raise ConnectionError(f"bad magic 0x{magic:08x}")
    if length > MAX_PAYLOAD:
        raise ConnectionError(f"frame length {length} exceeds {MAX_PAYLOAD}")
    return op, _recvn(f, length) if length else (op, b"")


def fixture_blob(layer: int, expert: int, nbytes: int = BLOB_BYTES) -> bytes:
    """Deterministic fixture bytes for (layer, expert): sha256 counter mode."""
    out = bytearray()
    counter = 0
    prefix = struct.pack("<QQ", layer, expert)
    while len(out) < nbytes:
        out += hashlib.sha256(prefix + struct.pack("<Q", counter)).digest()
        counter += 1
    return bytes(out[:nbytes])


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


class PageStore:
    """Serves fixture blobs for a fixed (layers, experts) grid."""

    def __init__(self, layers: int = LAYERS, experts: int = EXPERTS):
        self.layers = layers
        self.experts = experts
        self.gets = 0
        self.hits = 0
        self.misses = 0
        self.bytes_served = 0
        self._lock = threading.Lock()

    def in_range(self, layer: int, expert: int) -> bool:
        return 0 <= layer < self.layers and 0 <= expert < self.experts

    def get(self, layer: int, expert: int) -> bytes | None:
        with self._lock:
            self.gets += 1
        if not self.in_range(layer, expert):
            with self._lock:
                self.misses += 1
            return None
        blob = fixture_blob(layer, expert)
        with self._lock:
            self.hits += 1
            self.bytes_served += len(blob)
        return blob

    def stats(self):
        with self._lock:
            return self.gets, self.hits, self.misses, self.bytes_served


class _Handler(BaseRequestHandler):
    def handle(self):
        f = self.request.makefile("rb")
        store: PageStore = self.server.store
        try:
            while True:
                op, payload = decode_frame(f)
                if op == HELLO:
                    (ver,) = struct.unpack("<I", payload[:4])
                    if ver != VERSION:
                        self.request.sendall(encode_frame(ERROR, b"version"))
                        return
                    reply = struct.pack("<IQQQQ", VERSION, BLOB_BYTES, store.layers, store.experts, 0)
                    self.request.sendall(encode_frame(HELLO_REPLY, reply))
                elif op == GET:
                    layer, expert = struct.unpack("<QQ", payload[:16])
                    blob = store.get(layer, expert)
                    if blob is None:
                        self.request.sendall(encode_frame(MISS, struct.pack("<QQ", layer, expert)))
                    else:
                        self.request.sendall(
                            encode_frame(BLOB, struct.pack("<QQ", layer, expert) + sha256(blob) + blob)
                        )
                elif op == PREFETCH:
                    (n,) = struct.unpack("<I", payload[:4])
                    pairs = struct.unpack(f"<{n * 2}Q", payload[4:4 + n * 16])
                    for i in range(n):
                        store.get(pairs[2 * i], pairs[2 * i + 1])
                    self.request.sendall(encode_frame(ACK, struct.pack("<Q", 0)))
                elif op == STATS:
                    self.request.sendall(encode_frame(STATS_REPLY, struct.pack("<QQQQ", *store.stats())))
                elif op == BYE:
                    return
                else:
                    self.request.sendall(encode_frame(ERROR, b"unknown op"))
        except ConnectionError:
            return


class PageServer(ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, store: PageStore):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        self.store = store
        super().__init__(path, _Handler)


class PageClient:
    def __init__(self, path: str, name: str = "hip-engine"):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.f = self.sock.makefile("rb")
        payload = struct.pack("<I", VERSION) + struct.pack("<I", len(name)) + name.encode()
        self.sock.sendall(encode_frame(HELLO, payload))
        op, reply = decode_frame(self.f)
        if op != HELLO_REPLY:
            raise ConnectionError(f"expected HELLO_REPLY, got op {op}")
        ver, self.blob_bytes, self.layers, self.experts = struct.unpack("<IQQQ", reply[:28])
        self.vram_cached = struct.unpack("<Q", reply[28:36])[0] if len(reply) >= 36 else 0
        if ver != VERSION:
            raise ConnectionError(f"server version {ver} != {VERSION}")

    def get(self, layer: int, expert: int) -> bytes | None:
        self.sock.sendall(encode_frame(GET, struct.pack("<QQ", layer, expert)))
        op, payload = decode_frame(self.f)
        if op == MISS:
            return None
        if op == ERROR:
            raise ConnectionError(payload.decode(errors="replace"))
        if op != BLOB:
            raise ConnectionError(f"expected BLOB, got op {op}")
        rlayer, rexpert = struct.unpack("<QQ", payload[:16])
        digest, blob = payload[16:48], payload[48:]
        if (rlayer, rexpert) != (layer, expert):
            raise ConnectionError("blob address mismatch")
        if sha256(blob) != digest:
            raise ConnectionError("blob sha256 mismatch")
        return blob

    def prefetch(self, pairs) -> None:
        pairs = list(pairs)
        payload = struct.pack("<I", len(pairs))
        for layer, expert in pairs:
            payload += struct.pack("<QQ", layer, expert)
        self.sock.sendall(encode_frame(PREFETCH, payload))
        op, _ = decode_frame(self.f)
        if op != ACK:
            raise ConnectionError(f"expected ACK, got op {op}")

    def stats(self):
        self.sock.sendall(encode_frame(STATS, b""))
        op, payload = decode_frame(self.f)
        if op != STATS_REPLY:
            raise ConnectionError(f"expected STATS_REPLY, got op {op}")
        return struct.unpack("<QQQQ", payload)

    def close(self):
        try:
            self.sock.sendall(encode_frame(BYE, b""))
        except OSError:
            pass
        self.sock.close()

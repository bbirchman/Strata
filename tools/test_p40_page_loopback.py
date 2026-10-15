"""Loopback test for the P40 page protocol (no GPU, no network, no model files).

Starts a PageServer on a Unix socket in a temp dir, drives it with a
PageClient, and checks byte-exactness of full-geometry fixture blobs.

    python -m unittest tools.test_p40_page_loopback
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.p40_tier.page_proto import (  # noqa: E402
    BLOB_BYTES,
    PageClient,
    PageServer,
    PageStore,
    fixture_blob,
)


class PageLoopbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "p40.sock")
        self.store = PageStore(layers=48, experts=512)
        self.server = PageServer(self.path, self.store)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = PageClient(self.path, name="test")

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_hello_reports_geometry(self):
        self.assertEqual(self.client.blob_bytes, BLOB_BYTES)
        self.assertEqual((self.client.layers, self.client.experts), (48, 512))
        self.assertEqual(self.client.vram_cached, 0)
        self.assertTrue(self.client.daemon_holds(0, 0))
        self.assertTrue(self.client.daemon_holds(47, 511))
        self.assertFalse(self.client.daemon_holds(48, 0))

    def test_blobs_byte_exact(self):
        pairs = [(0, 0), (0, 511), (47, 511), (23, 256), (7, 42)]
        for layer, expert in pairs:
            got = self.client.get(layer, expert)
            self.assertIsNotNone(got)
            self.assertEqual(len(got), BLOB_BYTES)
            self.assertEqual(got, fixture_blob(layer, expert))

    def test_out_of_range_is_miss(self):
        self.assertIsNone(self.client.get(48, 0))
        self.assertIsNone(self.client.get(0, 512))
        self.assertIsNone(self.client.get(-1 & 0xFFFFFFFFFFFFFFFF, 0))

    def test_fetch_n_batch_byte_exact(self):
        from tools.p40_tier.page_proto import FETCH_N  # noqa: F401 (op wired)
        pairs = [(0, 0), (0, 511), (47, 511), (23, 256), (48, 0), (7, 42)]
        got = self.client.fetch_many(pairs)
        self.assertEqual(len(got), len(pairs))
        for (layer, expert, blob) in got:
            if layer >= 48:
                self.assertIsNone(blob)
            else:
                self.assertIsNotNone(blob)
                self.assertEqual(blob, fixture_blob(layer, expert))

    def test_prefetch_then_stats(self):
        pairs = [(i, i) for i in range(10)]
        self.client.prefetch(pairs)
        gets, hits, misses, served = self.client.stats()
        self.assertGreaterEqual(gets, 10)
        self.assertGreaterEqual(hits, 10)
        self.assertGreaterEqual(served, 10 * BLOB_BYTES)

    def test_round_trip_latency_us(self):
        self.client.get(1, 1)  # warmup
        n = 20
        t0 = time.perf_counter()
        for i in range(n):
            self.assertIsNotNone(self.client.get(2, i))
        dt = (time.perf_counter() - t0) / n
        print(f"\nloopback GET mean: {dt * 1e6:.0f} us/blob ({BLOB_BYTES / dt / 1e6:.1f} MB/s)")


if __name__ == "__main__":
    unittest.main()

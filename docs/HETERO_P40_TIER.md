# Hetero P40 tier: P40 VRAM as an expert tier behind a HIP main GPU

Status: prototype transport only. No engine changes yet. Default path untouched.

## Goal

Run IQ3_XXS / IQ2_XS at 256K on 7800XT (HIP, all compute) + Tesla P40 24GB
(CUDA, storage only) + 32GB RAM, with no RAM purchase. The P40 holds the
experts RAM cannot and serves their bytes; it never computes.

## Placement

VRAM-compute (7800XT) -> host RAM (hottest by profile rank) -> **P40 VRAM**
(next rank) -> SSD. The P40 sits *after* RAM: a host-RAM read costs one
PCIe hop (~0.06 ms/expert), a P40 round trip costs two (~0.17 ms), an SSD
miss ~0.38 ms (1.318 MiB blobs, PCIe3/4 and ~3.5 GB/s NVMe first-order
math, unmeasured). It converts SSD misses into 2x-faster P40 misses. It
does not turn the box into a 64 GB box.

## Protocol v1 (this branch)

`tools/p40_tier/page_proto.py`: Unix-socket transport, length-prefixed
frames `[magic|op|len|payload]` (hello/version handshake, GET -> BLOB with
sha256 or MISS, batched PREFETCH, STATS, per-piece size bound before
allocation). Framing shape lifted from upstream PR #1748's stage_link;
payload semantics are expert blobs, not activations.

Measured on this container (Python 3.13, Unix socket, sha256 per 1.3 MB
blob, no GPU): loopback GET mean 41,753 us/blob = 33.1 MB/s over 20
samples (`python -m unittest tools.test_p40_page_loopback`). That number
validates correctness only, not production throughput: the C++ path must
batch (`copy_blobs`) and use shared memory / pinned rows instead of one
Python round trip per blob.

## C++ tier (written, NOT yet compiled - no toolchain on this box)

- `include/strata/core/vendor_tier.hpp` + `src/core/vendor_tier.cpp`
  (wired into `strata_engine` in CMakeLists.txt): `VendorTierSource :
  ExpertSource` decorator. `copy_blob`/`copy_blobs` try the daemon first
  (sha256-verified, size-checked against `expert_layout().blob_bytes`),
  fall back per pair to the wrapped source; `blob()` stays on inner_
  (synchronous contract); `prefetch`/`warm` hint both; `pinned()` false
  (no SHM alias yet - bytes arrive by copy). Fail-open: any transport
  error only costs speed. Backend-neutral, Linux-gated with a stub
  elsewhere. One lock guards the socket (prompt stager is multi-threaded).
- Still to write: the CUDA daemon, the `--p40-tier SOCK` flag plumbing in
  `generate.cpp` (wrap `srcp` after the file/arena selection at
  ~line 5033), real pack-blob addressing in the daemon.

## Verification plan (per AGENTS.md)

- Default path byte-identical: flag off = last release behavior, checked
  by greedy-token comparison on a fixed prompt.
- Tier on: fixed expert sets (`--pcie-frac 0 --adapt-every 0
  --adapt-swaps 0`), one-token windows, compare greedy output vs the
  same-box single-GPU run; expect divergence across vendors, so the bar is
  coherence + sha-verified blobs, not bit-exactness.
- Build every backend touched (CUDA daemon, HIP engine) before review.
- Numbers reported with what they were measured on (GPUs, RAM, NVMe).

## Validation ladder (user PC: 7800XT + P40 + 32GB, CachyOS)

0. BEFORE ANY GPU RUN: stop llama-swap - `systemctl stop llama-swap`
   (it locks both GPUs). Re-check with `nvidia-smi` / `rocm-smi` that both
   cards are free before every rung below.
1. Coder 256K, single GPU (no tier): `--max-context 262144 --kv-resident
   32768 --resident-experts --kv int8` (+ `STRATA_DENSE_F32=1` on P40).
2. IQ2_XS 256K, single GPU, same flags.
3. XXS 128K, single GPU (setup default cap).
4. XXS 256K forced (`--max-context 262144`), single GPU.
5. Repeat 1-4 with `--p40-tier SOCK` (daemon on Ponce holding the cold
   tail): expect prefill faster-or-equal, decode equal-or-better, zero
   SSD reads in steady state; greedy-compare rung 4 vs rung 5 for
   coherence (divergence across vendors is expected, garbage is not).

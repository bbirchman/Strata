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

## C++ plan (not yet written)

- New CUDA daemon (own build, `STRATA_ENABLE_CUDA=ON
  -DSTRATA_EXPERIMENTAL_SM60=ON`): opens the pack, holds an ExpertCache on
  the P40, serves blobs over the socket. Storage-only `cudaMemcpy`, no
  sm_61 kernels.
- New `RemoteVendorSource : ExpertSource` in the HIP engine: `copy_blobs`
  fetches from the daemon into pinned rows; `prefetch`/`warm` batch ahead;
  `pinned()` false until SHM aliasing exists. Never in the default path:
  gated behind one opt-in flag (name TBD, e.g. `--p40-tier SOCK`).
- `ExpertDispatch` already carries `remote[3]` CUDA tiers; this tier is a
  *source* fallback (after RAM, before file), not a compute tier, so it
  plugs into the FileExpertSource fallback chain, not `RemoteExperts`.
- Backend guard stays: two processes, one runtime each. No change to
  `CMakeLists.txt` mutual exclusion.

## Verification plan (per AGENTS.md)

- Default path byte-identical: flag off = last release behavior, checked
  by greedy-token comparison on a fixed prompt.
- Tier on: fixed expert sets (`--pcie-frac 0 --adapt-every 0
  --adapt-swaps 0`), one-token windows, compare greedy output vs the
  same-box single-GPU run; expect divergence across vendors, so the bar is
  coherence + sha-verified blobs, not bit-exactness.
- Build every backend touched (CUDA daemon, HIP engine) before review.
- Numbers reported with what they were measured on (GPUs, RAM, NVMe).

## What remains

1. C++ `RemoteVendorSource` + daemon + flag plumbing in `generate.cpp`.
2. Real-blob addressing: (layer, expert) -> pack offsets (experts.bin /
   GGUF native layout), replacing the fixture synthesizer.
3. User-side validation on 7800XT + P40 + 32GB: resident-fit check,
   128K then 256K Coder / IQ2_XS / XXS runs with `--kv-resident 32768`.

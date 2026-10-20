// src/p40d/p40_daemon.cpp - het-p40-tier: the P40 page daemon.
//
// Serves expert blobs over a Unix socket (protocol v1, page_wire.hpp) to the
// HIP engine's VendorTierSource. Two modes:
//
//   --vram-gib 0    file-backed truth: bytes come straight from the pack's
//                   mapping (zero capacity gain; validates real-blob
//                   correctness end to end).
//   --vram-gib N    the coldest-ranked profile pairs that fit N GiB are
//                   pre-copied into P40 VRAM (plain cudaMalloc, storage
//                   only: NO kernels, NO sm_61 code paths) and served from
//                   there; anything else is a MISS the engine reads from
//                   RAM/file.
//
// Build (separate CUDA build dir, NOT the HIP engine dir). Example configure
// and build lines (fill in your toolkit path as needed):
//   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release
//     -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF
//     -DCMAKE_CUDA_ARCHITECTURES=61 -DSTRATA_EXPERIMENTAL_SM60=ON
//   cmake --build build-cuda --target p40_daemon -j$(nproc)
// Run with the P40 visible only to the daemon:
//   CUDA_VISIBLE_DEVICES=<p40-index> ./build-cuda/p40_daemon
//     --pack <model-dir> --profile <profile.bin> --socket /tmp/p40.sock
//     --vram-gib 20 [--gguf <native-shard>] [--device 0]
//
// Byte contract: blobs are FileExpertSource::blob_stable() bytes for the
// SAME --pack the engine opens (operator's job, like --expert-profile).
// Every BLOB carries sha256; the engine rejects size/address/hash mismatch.

#include "strata/core/expert_source.hpp"
#include "strata/core/page_wire.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cuda_runtime.h>

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <chrono>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace pw = strata::core::pagewire;

static std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&tt));
    return std::string(buf) + "." + std::to_string(ms.count()).substr(0,3);
}

namespace {

bool send_all(int fd, const uint8_t* p, size_t n) {
    while (n > 0) {
        ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return false;
        p += (size_t) w;
        n -= (size_t) w;
    }
    return true;
}
bool recv_all(int fd, uint8_t* p, size_t n) {
    while (n > 0) {
        ssize_t r = ::recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += (size_t) r;
        n -= (size_t) r;
    }
    return true;
}
bool send_frame(int fd, uint32_t op, const uint8_t* payload, size_t len) {
    if (len > pw::kMaxPayload) return false;
    pw::Header h{pw::kMagic, op, (uint64_t) len};
    return send_all(fd, (uint8_t*) &h, sizeof(h)) && (len == 0 || send_all(fd, payload, len));
}
bool recv_frame(int fd, uint32_t& op, std::vector<uint8_t>& payload) {
    pw::Header h{};
    if (!recv_all(fd, (uint8_t*) &h, sizeof(h))) return false;
    if (h.magic != pw::kMagic || h.len > pw::kMaxPayload) return false;
    op = h.op;
    payload.resize((size_t) h.len);
    return h.len == 0 || recv_all(fd, payload.data(), payload.size());
}

// STRP profile (tools/make_profile.py): "STRP" + 5xu32 LE
// (version, n_layer, n_expert, slots, n) + n x (u16 layer, u16 expert).
bool read_ranked(const std::string& path, int64_t& nl, int64_t& ne,
                 std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "profile: cannot open " + path;
        return false;
    }
    uint8_t head[24];
    if (std::fread(head, 1, sizeof(head), f) != sizeof(head) || std::memcmp(head, "STRP", 4) != 0) {
        std::fclose(f);
        err = "profile: not a STRP file";
        return false;
    }
    uint32_t v[5];
    std::memcpy(v, head + 4, 20);
    if (v[0] != 1) {
        std::fclose(f);
        err = "profile: unsupported version";
        return false;
    }
    nl = (int64_t) v[1];
    ne = (int64_t) v[2];
    uint32_t n = v[4];
    if (nl <= 0 || ne <= 0 || n == 0 || n > (uint32_t) (nl * ne)) {
        std::fclose(f);
        err = "profile: bad geometry";
        return false;
    }
    ranked.clear();
    ranked.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t e[4];
        if (std::fread(e, 1, 4, f) != 4) {
            std::fclose(f);
            err = "profile: truncated";
            return false;
        }
        int32_t l = (int32_t) (e[0] | ((uint32_t) e[1] << 8));
        int32_t x = (int32_t) (e[2] | ((uint32_t) e[3] << 8));
        if (l < 0 || l >= nl || x < 0 || x >= ne) {
            std::fclose(f);
            err = "profile: pair out of range";
            return false;
        }
        ranked.emplace_back(l, x);
    }
    std::fclose(f);
    return true;
}

struct Args {
    std::string pack, profile, socket, gguf;
    double vram_gib = 20.0;
    int device = 0;
};

bool parse_args(int argc, char** argv, Args& a, std::string& err) {
    auto next = [&](int& i) -> const char* {
        if (++i >= argc) return nullptr;
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        const char* v = nullptr;
        if (s == "--pack") v = next(i), a.pack = v ? v : "";
        else if (s == "--profile") v = next(i), a.profile = v ? v : "";
        else if (s == "--socket") v = next(i), a.socket = v ? v : "";
        else if (s == "--gguf") v = next(i), a.gguf = v ? v : "";
        else if (s == "--vram-gib") {
            v = next(i);
            if (v == nullptr) { err = "--vram-gib needs N"; return false; }
            a.vram_gib = std::atof(v);
        } else if (s == "--device") {
            v = next(i);
            if (v == nullptr) { err = "--device needs N"; return false; }
            a.device = std::atoi(v);
        } else {
            err = "unknown argument: " + s;
            return false;
        }
        if (v == nullptr && (s == "--pack" || s == "--profile" || s == "--socket" || s == "--gguf")) {
            err = s + " needs a value";
            return false;
        }
    }
    if (a.pack.empty() || a.profile.empty() || a.socket.empty()) {
        err = "need --pack DIR --profile FILE --socket PATH [--vram-gib N] [--gguf F] [--device N]";
        return false;
    }
    if (a.vram_gib < 0.0) {
        err = "--vram-gib needs N >= 0";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    std::string err;
    if (!parse_args(argc, argv, a, err)) {
        std::fprintf(stderr, "p40_daemon: [%s] %s\n", timestamp().c_str(), err.c_str());
        return 2;
    }
    if (::cudaSetDevice(a.device) != cudaSuccess) {
        std::fprintf(stderr, "p40_daemon: [%s] cudaSetDevice(%d) failed: %s\n", timestamp().c_str(), a.device,
                     ::cudaGetErrorString(::cudaGetLastError()));
        return 1;
    }
    int64_t nl = 0, ne = 0;
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (!read_ranked(a.profile, nl, ne, ranked, err)) {
        std::fprintf(stderr, "p40_daemon: [%s] %s\n", timestamp().c_str(), err.c_str());
        return 1;
    }
    strata::core::FileExpertSource src;
    if (!a.gguf.empty()) src.set_gguf(a.gguf);
    // The process-wide layout comes from the pack's index.txt (generate.cpp
    // does this before opening any source); without it the global still
    // says canonical 48x512 and a pruned pack (Coder: 256/layer) refuses.
    if (!strata::kernels::cpu::expert_layout_load(a.pack, nl, ne, err)) {
        std::fprintf(stderr, "p40_daemon: [%s] %s\n", timestamp().c_str(), err.c_str());
        return 1;
    }
    if (!src.open(a.pack, nl, ne, err)) {
        std::fprintf(stderr, "p40_daemon: [%s] %s\n", timestamp().c_str(), err.c_str());
        return 1;
    }
    const auto& pl = strata::kernels::cpu::expert_layout();
    uint64_t max_blob = pl.max_blob;

    // Coldest-ranked tail that fits the VRAM budget. The engine holds the
    // hot head in its VRAM/RAM; overlap is harmless (same bytes).
    struct Held {
        int32_t l, e;
        uint64_t off, bytes;
    };
    std::vector<Held> held;
    std::vector<int32_t> slot((size_t) (nl * ne), -1);
    uint64_t budget = (uint64_t) (a.vram_gib * 1073741824.0);
    uint64_t used = 0;
    if (budget > 0) {
        for (int64_t i = (int64_t) ranked.size() - 1; i >= 0; --i) {
            int32_t l = ranked[(size_t) i].first, e = ranked[(size_t) i].second;
            uint64_t b = pl.blob_bytes(l);
            if (used + b > budget) continue;
            slot[(size_t) (l * ne + e)] = (int32_t) held.size();
            held.push_back({l, e, used, b});
            used += b;
        }
    }
    uint8_t* dev = nullptr;
    if (!held.empty()) {
        if (::cudaMalloc(&dev, (size_t) used) != cudaSuccess) {
            std::fprintf(stderr, "p40_daemon: [%s] cudaMalloc(%.2f GiB) failed: %s\n", timestamp().c_str(),
                         (double) used / 1073741824.0, ::cudaGetErrorString(::cudaGetLastError()));
            return 1;
        }
        auto t0 = std::chrono::steady_clock::now();
        uint64_t done = 0;
        for (const auto& h : held) {
            const uint8_t* b = src.blob_stable(h.l, h.e);
            if (b == nullptr) {
                std::fprintf(stderr, "p40_daemon: [%s] pack read failed at (%d,%d)\n", timestamp().c_str(), h.l, h.e);
                return 1;
            }
            if (::cudaMemcpy((uint8_t*) dev + h.off, b, (size_t) h.bytes, cudaMemcpyHostToDevice) !=
                cudaSuccess) {
                std::fprintf(stderr, "p40_daemon: [%s] H2D fill failed: %s\n", timestamp().c_str(),
                             ::cudaGetErrorString(::cudaGetLastError()));
                return 1;
            }
            done += h.bytes;
            if ((done >> 30) != ((done - h.bytes) >> 30))
                std::fprintf(stderr, "p40_daemon: [%s] filled %.2f GiB\r", timestamp().c_str(), (double) done / 1073741824.0);
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr,
                     "\np40_daemon: [%s] VRAM cache: %zu experts, %.2f GiB at %.2f GiB/s (--vram-gib %.1f)\n",
                     timestamp().c_str(), held.size(), (double) used / 1073741824.0, s > 0 ? (double) used / 1073741824.0 / s : 0,
                     a.vram_gib);
    std::fprintf(stderr, "p40_daemon: [%s] geometry %lld layers x %lld experts, slot table %zu; held head (%d,%d) tail (%d,%d)\n",
                 timestamp().c_str(), (long long) nl, (long long) ne, slot.size(), held.empty() ? -1 : held.front().l,
                 held.empty() ? -1 : held.front().e, held.empty() ? -1 : held.back().l,
                 held.empty() ? -1 : held.back().e);
    } else {
        std::fprintf(stderr, "p40_daemon: [%s] file-backed mode (--vram-gib 0): serving pack bytes, no VRAM cache\n", timestamp().c_str());
    }

    uint8_t* stage = nullptr;
    if (::cudaHostAlloc(&stage, (size_t) max_blob, cudaHostAllocDefault) != cudaSuccess) {
        std::fprintf(stderr, "p40_daemon: [%s] pinned staging alloc failed: %s\n", timestamp().c_str(),
                     ::cudaGetErrorString(::cudaGetLastError()));
        return 1;
    }
    int srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) {
        std::fprintf(stderr, "p40_daemon: [%s] socket() failed\n", timestamp().c_str());
        return 1;
    }
    ::unlink(a.socket.c_str());
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (a.socket.size() >= sizeof(addr.sun_path)) {
        std::fprintf(stderr, "p40_daemon: [%s] socket path too long\n", timestamp().c_str());
        return 1;
    }
    std::memcpy(addr.sun_path, a.socket.c_str(), a.socket.size() + 1);
    if (::bind(srv, (sockaddr*) &addr, sizeof(addr)) != 0 || ::listen(srv, 4) != 0) {
        std::fprintf(stderr, "p40_daemon: [%s] bind/listen failed\n", timestamp().c_str());
        return 1;
    }
    // Bulk BLOBs (2+ MB) must never wedge against a pausing peer: 16 MB
    // buffers absorb whole frames either way; 30 s I/O deadlines stop a
    // wedged client from pinning the single serving thread forever.
    int bigbuf = 16 << 20;
    ::setsockopt(srv, SOL_SOCKET, SO_SNDBUF, &bigbuf, sizeof(bigbuf));
    ::setsockopt(srv, SOL_SOCKET, SO_RCVBUF, &bigbuf, sizeof(bigbuf));
    std::fprintf(stderr, "p40_daemon: [%s] pack %s, profile %s (%lld pairs), socket %s, device %d\n",
                 timestamp().c_str(), a.pack.c_str(), a.profile.c_str(), (long long) ranked.size(), a.socket.c_str(),
                 a.device);
    uint64_t gets = 0, hits = 0, misses = 0, served = 0, d2h_us = 0;
    long long trace_n = 0;
    const bool trace = [] {
        const char* e = std::getenv("STRATA_P40_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    for (;;) {
        int fd = ::accept(srv, nullptr, nullptr);
        if (fd < 0) break;
        std::fprintf(stderr, "p40_daemon: [%s] client connected\n", timestamp().c_str());
        ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bigbuf, sizeof(bigbuf));
        ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bigbuf, sizeof(bigbuf));
        timeval dtv{3600, 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &dtv, sizeof(dtv));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &dtv, sizeof(dtv));
        for (;;) {
            uint32_t op = 0;
            std::vector<uint8_t> req;
            if (!recv_frame(fd, op, req)) break;
            if (op == pw::HELLO) {
                if (req.size() < 4 || req[0] != 1) {
                    const char* m = "version";
                    send_frame(fd, (uint32_t) pw::ERROR, (const uint8_t*) m, 7);
                    break;
                }
                std::vector<uint8_t> rep;
                auto pu32 = [&](uint32_t x) {
                    rep.push_back((uint8_t) x);
                    rep.push_back((uint8_t) (x >> 8));
                    rep.push_back((uint8_t) (x >> 16));
                    rep.push_back((uint8_t) (x >> 24));
                };
                auto pu64 = [&](uint64_t x) {
                    for (int i = 0; i < 8; ++i) rep.push_back((uint8_t) (x >> (8 * i)));
                };
                pu32(1);
                pu64(max_blob);
                pu64((uint64_t) nl);
                pu64((uint64_t) ne);
                pu64(used);  // HELLO_REPLY field 5: VRAM-cached bytes (0 = file mode)
                // HELLO_REPLY field 6: held set, one bit per pair row-major.
                // Lets the engine skip hot pairs without a round trip.
                {
                    size_t mapbytes = (size_t) (((uint64_t) nl * (uint64_t) ne + 7) / 8);
                    size_t base = rep.size();
                    rep.resize(base + mapbytes, 0);
                    for (int64_t l = 0; l < nl; ++l)
                        for (int64_t e = 0; e < ne; ++e) {
                            uint64_t i = (uint64_t) l * (uint64_t) ne + (uint64_t) e;
                            if (slot[(size_t) i] >= 0) rep[base + (i >> 3)] |= (uint8_t) (1u << (i & 7));
                        }
                }
                if (!send_frame(fd, (uint32_t) pw::HELLO_REPLY, rep.data(), rep.size())) break;
                continue;  // het-p40-tier fix: HELLO must not fall into the GET/else chain (stale ERROR)
            }

            // One pair lookup + reply. True = stream in sync, keep serving;
            // false = transport broke, drop the client.
            auto serve_pair = [&](int64_t l, int64_t e) -> bool {
                ++gets;
                if (gets % 1000 == 0)
                    std::fprintf(stderr,
                                 "p40_daemon: [%s] progress: %llu gets %llu hits %llu misses %.2f GiB served, mean D2H %llu us\n",
                                 timestamp().c_str(), (unsigned long long) gets, (unsigned long long) hits,
                                 (unsigned long long) misses, (double) served / 1073741824.0,
                                 hits > 0 ? d2h_us / hits : 0);
                int32_t s = (l < 0 || e < 0 || l >= nl || e >= ne) ? -1 : slot[(size_t) (l * ne + e)];
                const uint8_t* bytes = nullptr;
                uint64_t nb = 0;
                if (s >= 0 && dev != nullptr) {
                    const auto d2h0 = std::chrono::steady_clock::now();
                    bool d2h_ok = ::cudaMemcpy(stage, (uint8_t*) dev + held[(size_t) s].off,
                                               (size_t) held[(size_t) s].bytes,
                                               cudaMemcpyDeviceToHost) == cudaSuccess;
                    d2h_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - d2h0).count();
                    if (d2h_ok) {
                        bytes = stage;
                        nb = held[(size_t) s].bytes;
                    }
                } else if (s < 0 && dev == nullptr) {
                    // File-backed mode: every in-range pair is servable.
                    if (l >= 0 && e >= 0 && l < nl && e < ne) {
                        bytes = src.blob_stable((int32_t) l, (int32_t) e);
                        nb = bytes != nullptr ? pl.blob_bytes((int32_t) l) : 0;
                    }
                }
                if (bytes == nullptr || nb == 0) {
                    ++misses;
                    if (trace && trace_n < 200) {
                        std::fprintf(stderr, "p40d: [%s] GET (%lld,%lld) -> MISS\n", timestamp().c_str(), (long long) l,
                                     (long long) e);
                        ++trace_n;
                    }
                    uint8_t m[16];
                    std::memcpy(m, &l, 8);
                    std::memcpy(m + 8, &e, 8);
                    if (!send_frame(fd, (uint32_t) pw::MISS, m, sizeof(m))) return false;
                    return true;
                }
                ++hits;
                served += nb;
                if (trace && trace_n < 200) {
                    std::fprintf(stderr, "p40d: [%s] GET (%lld,%lld) -> BLOB %llu B\n", timestamp().c_str(), (long long) l,
                                 (long long) e, (unsigned long long) nb);
                    ++trace_n;
                }
                pw::Sha256 hsh;
                static bool verify = [] {
                    const char* e = std::getenv("STRATA_P40_VERIFY");
                    return e != nullptr && e[0] == '1';
                }();
                // Content proof moved to open-time sentinels (see tier):
                // hashing every 2 MB blob here costs more CPU than the PCIe
                // transfer. Zeros unless both ends opt into VERIFY=1.
                std::array<uint8_t, 32> dg{};
                if (verify) {
                    hsh.update(bytes, (size_t) nb);
                    dg = hsh.final();
                }
                // One frame: len = 48 + nb, payload = meta + bytes, exactly
                // what VendorTierSource::fetch_one reads (and what the
                // Python page server sends).
                std::vector<uint8_t> frame;
                frame.reserve(48 + (size_t) nb);
                for (int i = 0; i < 8; ++i) frame.push_back(((uint8_t*) &l)[i]);
                for (int i = 0; i < 8; ++i) frame.push_back(((uint8_t*) &e)[i]);
                frame.insert(frame.end(), dg.begin(), dg.end());
                frame.insert(frame.end(), bytes, bytes + (size_t) nb);
                if (!send_frame(fd, (uint32_t) pw::BLOB, frame.data(), frame.size())) return false;
                return true;
            };

            // het-p40-tier COMPUTE_N: grouped expert compute on daemon VRAM.
            // v1: single token; pairs grouped by layer (uniform blob_bytes per
            // kernel call); blobs gathered D2D into a uniform arena; replies
            // in request order (COMPUTE_REPLY or MISS each). Any transport or
            // CUDA failure -> false (caller breaks: stream stays synced, the
            // engine falls back to GETs). MISS legs drain via engine discipline.
            auto serve_compute = [&](const std::vector<uint8_t>& rq) -> bool {
                // v2: native_expert_grouped (native IQ layouts) + q8_1 input.
                // Req: {u32 n, u32 layer, u32 n_embd, u32 n_ff, u32 gu, u32 d,
                //   n x (u64 l, u64 e; l must == layer), q8_1 act ((n_embd/32)*36 B)}.
                // Replies per pair in order (COMPUTE_REPLY or MISS). Validation
                // trouble -> ERROR + stay connected (engine falls back to GETs);
                // CUDA trouble -> false (drop; stream may be poisoned).
                auto errstay = [&](const char* why) -> bool {
                    std::fprintf(stderr, "p40d: COMPUTE reject: %s (rq=%zu)\n", why, rq.size());
                    const char* m = "compute unsupported";
                    return send_frame(fd, (uint32_t)pw::ERROR, (const uint8_t*)m, 17); };
                if (rq.size() < 24) return false;
                uint32_t n = 0, layer = 0, n_embd = 0, n_ff = 0, gu = 0, dt = 0;
                std::memcpy(&n, rq.data(), 4); std::memcpy(&layer, rq.data() + 4, 4);
                std::memcpy(&n_embd, rq.data() + 8, 4); std::memcpy(&n_ff, rq.data() + 12, 4);
                std::memcpy(&gu, rq.data() + 16, 4); std::memcpy(&dt, rq.data() + 20, 4);
                if (n == 0 || n > 64 || n_embd == 0 || n_embd > 8192 || (n_embd % 32) != 0 ||
                    n_ff == 0 || n_ff > 32768 || gu > 64 || dt > 64) return errstay("ranges");
                size_t actb = (size_t)(n_embd / 32) * 36;
                if (rq.size() != 24 + (size_t)n * 16 + actb) return errstay("size");
                if (!strata::kernels::native_expert_supported((int)gu, (int)dt, (int64_t)n_embd,
                                                              (int64_t)n_ff)) return errstay("supported");
                strata::kernels::NativeExpertLayout L =
                    strata::kernels::native_expert_layout((int)gu, (int)dt, (int64_t)n_embd, (int64_t)n_ff);
                {
                    strata::kernels::cpu::NativeFmt F;
                    std::string ferr;
                    bool fok = strata::kernels::cpu::native_fmt((int)gu, (int)dt, (int64_t)n_embd,
                                                                (int64_t)n_ff, F, ferr);
                    static bool once2 = false;
                    if (!once2) {
                        once2 = true;
                        std::fprintf(stderr, "p40d: fmt gu_row=%zu/%zu d_row=%zu/%zu up=%zu/%zu down=%zu/%zu bytes=%zu/%zu %s\n",
                                     L.gu_row, fok ? F.gu_row : 0, L.d_row, fok ? F.d_row : 0,
                                     (size_t)0, fok ? F.up_off : 0, (size_t)0, fok ? F.down_off : 0,
                                     L.bytes, fok ? F.bytes : 0, fok ? "FOK" : ferr.c_str());
                    }
                }
                {
                    uint64_t bb = pl.blob_bytes((int64_t)layer);
                    static bool once = false;
                    if (!once) {
                        once = true;
                        std::fprintf(stderr, "p40d: layout bytes=%zu blob_bytes=%llu gu=%u d=%u embd=%u ff=%u\n",
                                     L.bytes, (unsigned long long)bb, gu, dt, n_embd, n_ff);
                    }
                }
                static cudaStream_t cs = nullptr;
                static uint8_t *xq_d = nullptr, *out_d = nullptr;
                static unsigned long long* gp_d = nullptr;
                static int32_t *gs_d = nullptr, *ng_d = nullptr, *ed_d = nullptr, *et_d = nullptr;
                static void* scratch = nullptr; static size_t scratch_n = 0;
                if (cs == nullptr) {
                    if (::cudaStreamCreate(&cs) != cudaSuccess) return false;
                    if (::cudaMalloc(&xq_d, (size_t)(8192 / 32) * 36) != cudaSuccess) return false;
                    if (::cudaMalloc(&out_d, (size_t)64 * (size_t)8192 * 4) != cudaSuccess) return false;
                    if (::cudaMalloc(&gp_d, (size_t)64 * 8) != cudaSuccess) return false;
                    if (::cudaMalloc(&gs_d, (size_t)65 * 4) != cudaSuccess) return false;
                    if (::cudaMalloc(&ng_d, (size_t)4) != cudaSuccess) return false;
                    if (::cudaMalloc(&ed_d, (size_t)64 * 4) != cudaSuccess) return false;
                    if (::cudaMalloc(&et_d, (size_t)64 * 4) != cudaSuccess) return false;
                }
                size_t need = strata::kernels::native_expert_scratch_bytes(64, (int64_t)n_ff);
                if (need > scratch_n) {
                    if (scratch != nullptr) ::cudaFree(scratch);
                    if (::cudaMalloc(&scratch, need) != cudaSuccess) { scratch = nullptr; scratch_n = 0; return false; }
                    scratch_n = need;
                }
                if (::cudaMemcpyAsync(xq_d, rq.data() + 24 + (size_t)n * 16, actb, cudaMemcpyHostToDevice, cs) !=
                    cudaSuccess) return false;
                std::vector<unsigned long long> gp(n); std::vector<int32_t> ed(n), et(n, 0);
                std::vector<int64_t> rl(n, 0), re(n, 0); std::vector<char> is_miss(n, 0);
                size_t m = 0;
                for (size_t k = 0; k < n; ++k) {
                    int64_t l = 0, e = 0;
                    std::memcpy(&l, rq.data() + 24 + k * 16, 8);
                    std::memcpy(&e, rq.data() + 24 + k * 16 + 8, 8);
                    rl[k] = l; re[k] = e;
                    if (l != (int64_t)layer) return errstay("layer");
                    int32_t sl = (l < 0 || e < 0 || l >= nl || e >= ne) ? -1 : slot[(size_t)(l * ne + e)];
                    if (sl < 0) { is_miss[k] = 1; continue; }
                    gp[m] = (unsigned long long)(dev + held[(size_t)sl].off);
                    ed[m] = (int32_t)m; ++m;
                }
                if (m > 0) {
                    int32_t gs[2] = {0, (int32_t)m}, ng = 1;
                    if (::cudaMemcpyAsync(gp_d, gp.data(), m * 8, cudaMemcpyHostToDevice, cs) != cudaSuccess) return false;
                    if (::cudaMemcpyAsync(gs_d, gs, 8, cudaMemcpyHostToDevice, cs) != cudaSuccess) return false;
                    if (::cudaMemcpyAsync(ng_d, &ng, 4, cudaMemcpyHostToDevice, cs) != cudaSuccess) return false;
                    if (::cudaMemcpyAsync(ed_d, ed.data(), m * 4, cudaMemcpyHostToDevice, cs) != cudaSuccess) return false;
                    if (::cudaMemcpyAsync(et_d, et.data(), m * 4, cudaMemcpyHostToDevice, cs) != cudaSuccess) return false;
                    strata::kernels::native_expert_grouped(
                        L, gp_d, gs_d, ng_d, ed_d, et_d, 1, (int64_t)m, xq_d, scratch,
                        (float*)out_d, (void*)cs, 0);
                    if (::cudaGetLastError() != cudaSuccess) return false;
                    if (::cudaStreamSynchronize(cs) != cudaSuccess) return false;
                }
                std::vector<uint8_t> outh(n * (size_t)n_embd * 4, 0);
                if (m > 0 &&
                    ::cudaMemcpy(outh.data(), out_d, m * (size_t)n_embd * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
                    return false;
                for (size_t k = 0, r = 0; k < n; ++k) {
                    if (is_miss[k]) {
                        ++misses;
                        uint8_t mm[16];
                        std::memcpy(mm, &rl[k], 8); std::memcpy(mm + 8, &re[k], 8);
                        if (!send_frame(fd, (uint32_t)pw::MISS, mm, sizeof(mm))) return false;
                        continue;
                    }
                    ++gets; ++hits;
                    std::vector<uint8_t> fr;
                    fr.reserve(16 + (size_t)n_embd * 4);
                    for (int b = 0; b < 8; ++b) fr.push_back(((uint8_t*)&rl[k])[b]);
                    for (int b = 0; b < 8; ++b) fr.push_back(((uint8_t*)&re[k])[b]);
                    fr.insert(fr.end(), outh.data() + r * (size_t)n_embd * 4,
                              outh.data() + (r + 1) * (size_t)n_embd * 4);
                    ++r;
                    if (trace && trace_n < 200) {
                        std::fprintf(stderr, "p40d: [%s] COMPUTE (%lld,%lld) -> %llu B out\n",
                                     timestamp().c_str(), (long long)rl[k], (long long)re[k],
                                     (unsigned long long)((size_t)n_embd * 4));
                        ++trace_n;
                    }
                    if (!send_frame(fd, (uint32_t)pw::COMPUTE_REPLY, fr.data(), fr.size())) return false;
                }
                return true;
            };
            if (op == pw::GET) {
                if (req.size() != 16) break;
                int64_t l = 0, e = 0;
                std::memcpy(&l, req.data(), 8);
                std::memcpy(&e, req.data() + 8, 8);
                if (!serve_pair(l, e)) break;
            } else if (op == pw::FETCH_N) {
                if (req.size() < 4) break;
                uint32_t n = 0;
                std::memcpy(&n, req.data(), 4);
                if (n == 0 || n > 64 || req.size() != 4 + (size_t) n * 16) break;
                bool ok = true;
                for (uint32_t i = 0; ok && i < n; ++i) {
                    int64_t l = 0, e = 0;
                    std::memcpy(&l, req.data() + 4 + (size_t) i * 16, 8);
                    std::memcpy(&e, req.data() + 4 + (size_t) i * 16 + 8, 8);
                    ok = serve_pair(l, e);
                }
                if (!ok) break;
            } else if (op == pw::PREFETCH) {
                // Bytes are resident by construction (VRAM) or pageable
                // (file mode): nothing to stage, ACK keeps the stream.
                uint8_t z[8] = {};
                if (!send_frame(fd, (uint32_t) pw::ACK, z, sizeof(z))) break;
            } else if (op == pw::STATS) {
                std::vector<uint8_t> rep;
                for (uint64_t x : {gets, hits, misses, served})
                    for (int i = 0; i < 8; ++i) rep.push_back((uint8_t) (x >> (8 * i)));
                if (!send_frame(fd, (uint32_t) pw::STATS_REPLY, rep.data(), rep.size())) break;
            } else if (op == pw::BYE) {
                break;
            } else if (op == pw::COMPUTE_N) {
                if (!serve_compute(req)) break;
            } else {
                const char* m = "unknown op";
                if (!send_frame(fd, (uint32_t) pw::ERROR, (const uint8_t*) m, 10)) break;
            }
        }
        ::close(fd);
        std::fprintf(stderr, "p40_daemon: [%s] client done: %llu gets %llu hits %llu misses %.2f GiB served\n",
                     timestamp().c_str(), (unsigned long long) gets, (unsigned long long) hits, (unsigned long long) misses,
                     (double) served / 1073741824.0);
    }
    return 0;
}

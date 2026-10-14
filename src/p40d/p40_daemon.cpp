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

#include <cuda_runtime.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace pw = strata::core::pagewire;

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
        std::fprintf(stderr, "p40_daemon: %s\n", err.c_str());
        return 2;
    }
    if (::cudaSetDevice(a.device) != cudaSuccess) {
        std::fprintf(stderr, "p40_daemon: cudaSetDevice(%d) failed: %s\n", a.device,
                     ::cudaGetErrorString(::cudaGetLastError()));
        return 1;
    }
    int64_t nl = 0, ne = 0;
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (!read_ranked(a.profile, nl, ne, ranked, err)) {
        std::fprintf(stderr, "p40_daemon: %s\n", err.c_str());
        return 1;
    }
    strata::core::FileExpertSource src;
    if (!a.gguf.empty()) src.set_gguf(a.gguf);
    // The process-wide layout comes from the pack's index.txt (generate.cpp
    // does this before opening any source); without it the global still
    // says canonical 48x512 and a pruned pack (Coder: 256/layer) refuses.
    if (!strata::kernels::cpu::expert_layout_load(a.pack, nl, ne, err)) {
        std::fprintf(stderr, "p40_daemon: %s\n", err.c_str());
        return 1;
    }
    if (!src.open(a.pack, nl, ne, err)) {
        std::fprintf(stderr, "p40_daemon: %s\n", err.c_str());
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
            std::fprintf(stderr, "p40_daemon: cudaMalloc(%.2f GiB) failed: %s\n",
                         (double) used / 1073741824.0, ::cudaGetErrorString(::cudaGetLastError()));
            return 1;
        }
        auto t0 = std::chrono::steady_clock::now();
        uint64_t done = 0;
        for (const auto& h : held) {
            const uint8_t* b = src.blob_stable(h.l, h.e);
            if (b == nullptr) {
                std::fprintf(stderr, "p40_daemon: pack read failed at (%d,%d)\n", h.l, h.e);
                return 1;
            }
            if (::cudaMemcpy((uint8_t*) dev + h.off, b, (size_t) h.bytes, cudaMemcpyHostToDevice) !=
                cudaSuccess) {
                std::fprintf(stderr, "p40_daemon: H2D fill failed: %s\n",
                             ::cudaGetErrorString(::cudaGetLastError()));
                return 1;
            }
            done += h.bytes;
            if ((done >> 30) != ((done - h.bytes) >> 30))
                std::fprintf(stderr, "p40_daemon: filled %.2f GiB\r", (double) done / 1073741824.0);
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr,
                     "\np40_daemon: VRAM cache: %zu experts, %.2f GiB at %.2f GiB/s (--vram-gib %.1f)\n",
                     held.size(), (double) used / 1073741824.0, s > 0 ? (double) used / 1073741824.0 / s : 0,
                     a.vram_gib);
    } else {
        std::fprintf(stderr, "p40_daemon: file-backed mode (--vram-gib 0): serving pack bytes, no VRAM cache\n");
    }

    uint8_t* stage = nullptr;
    if (::cudaHostAlloc(&stage, (size_t) max_blob, cudaHostAllocDefault) != cudaSuccess) {
        std::fprintf(stderr, "p40_daemon: pinned staging alloc failed: %s\n",
                     ::cudaGetErrorString(::cudaGetLastError()));
        return 1;
    }
    int srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) {
        std::fprintf(stderr, "p40_daemon: socket() failed\n");
        return 1;
    }
    ::unlink(a.socket.c_str());
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (a.socket.size() >= sizeof(addr.sun_path)) {
        std::fprintf(stderr, "p40_daemon: socket path too long\n");
        return 1;
    }
    std::memcpy(addr.sun_path, a.socket.c_str(), a.socket.size() + 1);
    if (::bind(srv, (sockaddr*) &addr, sizeof(addr)) != 0 || ::listen(srv, 4) != 0) {
        std::fprintf(stderr, "p40_daemon: bind/listen failed\n");
        return 1;
    }
    std::fprintf(stderr, "p40_daemon: pack %s, profile %s (%lld pairs), socket %s, device %d\n",
                 a.pack.c_str(), a.profile.c_str(), (long long) ranked.size(), a.socket.c_str(),
                 a.device);
    uint64_t gets = 0, hits = 0, misses = 0, served = 0;
    for (;;) {
        int fd = ::accept(srv, nullptr, nullptr);
        if (fd < 0) break;
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
                if (!send_frame(fd, (uint32_t) pw::HELLO_REPLY, rep.data(), rep.size())) break;
            } else if (op == pw::GET) {
                ++gets;
                if (req.size() != 16) break;
                int64_t l = 0, e = 0;
                std::memcpy(&l, req.data(), 8);
                std::memcpy(&e, req.data() + 8, 8);
                int32_t s = (l < 0 || e < 0 || l >= nl || e >= ne) ? -1 : slot[(size_t) (l * ne + e)];
                const uint8_t* bytes = nullptr;
                uint64_t nb = 0;
                if (s >= 0 && dev != nullptr) {
                    if (::cudaMemcpy(stage, (uint8_t*) dev + held[(size_t) s].off,
                                     (size_t) held[(size_t) s].bytes,
                                     cudaMemcpyDeviceToHost) == cudaSuccess) {
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
                    uint8_t m[16];
                    std::memcpy(m, &l, 8);
                    std::memcpy(m + 8, &e, 8);
                    if (!send_frame(fd, (uint32_t) pw::MISS, m, sizeof(m))) break;
                    continue;
                }
                ++hits;
                served += nb;
                pw::Sha256 hsh;
                hsh.update(bytes, (size_t) nb);
                auto dg = hsh.final();
                // One frame: len = 48 + nb, payload = meta + bytes, exactly
                // what VendorTierSource::fetch_one reads (and what the
                // Python page server sends).
                std::vector<uint8_t> frame;
                frame.reserve(48 + (size_t) nb);
                for (int i = 0; i < 8; ++i) frame.push_back(((uint8_t*) &l)[i]);
                for (int i = 0; i < 8; ++i) frame.push_back(((uint8_t*) &e)[i]);
                frame.insert(frame.end(), dg.begin(), dg.end());
                frame.insert(frame.end(), bytes, bytes + (size_t) nb);
                if (!send_frame(fd, (uint32_t) pw::BLOB, frame.data(), frame.size())) break;
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
            } else {
                const char* m = "unknown op";
                if (!send_frame(fd, (uint32_t) pw::ERROR, (const uint8_t*) m, 10)) break;
            }
        }
        ::close(fd);
        std::fprintf(stderr, "p40_daemon: client done: %llu gets %llu hits %llu misses %.2f GiB served\n",
                     (unsigned long long) gets, (unsigned long long) hits, (unsigned long long) misses,
                     (double) served / 1073741824.0);
    }
    return 0;
}

// src/core/vendor_tier.cpp - het-p40-tier: Unix-socket client for the P40
// page daemon. POSIX only (the tier is a Linux-server feature; the file
// compiles nowhere else - see the guard below).
//
// Fail-open everywhere: any transport error, MISS, size or hash mismatch
// returns false and the caller uses the wrapped source. A broken daemon
// costs speed, never correctness.

#include "strata/core/vendor_tier.hpp"

#if defined(__linux__)

#include "strata/kernels/cpu/expert_layout.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/select.h>
#include <sys/time.h>
#include <vector>

namespace strata::core {
namespace {

// ---- minimal SHA-256 (public-domain shape, no new link deps) ----
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t total = 0;
    uint8_t buf[64] = {};
    size_t buflen = 0;
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t) p[4 * i] << 24) | ((uint32_t) p[4 * i + 1] << 16) |
                   ((uint32_t) p[4 * i + 2] << 8) | (uint32_t) p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const uint8_t* p, size_t n) {
        total += n;
        while (n > 0) {
            size_t take = 64 - buflen < n ? 64 - buflen : n;
            std::memcpy(buf + buflen, p, take);
            buflen += take; p += take; n -= take;
            if (buflen == 64) { block(buf); buflen = 0; }
        }
    }
    std::array<uint8_t, 32> final() {
        uint64_t bits = total * 8;
        uint8_t one = 0x80;
        update(&one, 1);
        uint8_t zero = 0;
        while (buflen != 56) update(&zero, 1);
        uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) lenb[i] = (uint8_t) (bits >> (56 - 8 * i));
        update(lenb, 8);
        std::array<uint8_t, 32> out;
        for (int i = 0; i < 8; ++i) {
            out[4 * i] = (uint8_t) (h[i] >> 24); out[4 * i + 1] = (uint8_t) (h[i] >> 16);
            out[4 * i + 2] = (uint8_t) (h[i] >> 8); out[4 * i + 3] = (uint8_t) h[i];
        }
        return out;
    }
};

// ---- framing (must match tools/p40_tier/page_proto.py v1) ----
constexpr uint32_t kMagic = 0x50343054;
constexpr uint32_t kVersion = 1;
constexpr uint64_t kMaxPayload = 8ull << 20;
enum Op : uint32_t { HELLO = 1, HELLO_REPLY = 2, GET = 3, BLOB = 4, MISS = 5,
                     PREFETCH = 6, ACK = 7, STATS = 8, STATS_REPLY = 9, BYE = 10, ERROR = 11 };

#pragma pack(push, 1)
struct Header { uint32_t magic, op; uint64_t len; };
#pragma pack(pop)
static_assert(sizeof(Header) == 16, "frame header is 16 bytes");

bool send_all(int fd, const uint8_t* p, size_t n) {
    while (n > 0) {
        ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return false;
        p += (size_t) w; n -= (size_t) w;
    }
    return true;
}
bool recv_all(int fd, uint8_t* p, size_t n) {
    while (n > 0) {
        ssize_t r = ::recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += (size_t) r; n -= (size_t) r;
    }
    return true;
}
void put_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t) x); v.push_back((uint8_t) (x >> 8));
    v.push_back((uint8_t) (x >> 16)); v.push_back((uint8_t) (x >> 24));
}
void put_u64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back((uint8_t) (x >> (8 * i)));
}

}  // namespace

VendorTierSource::~VendorTierSource() { close(); }

bool VendorTierSource::open(const std::string& socket_path, ExpertSource* inner,
                            int64_t layers, int64_t experts, std::string& err) {
    close();
    if (inner == nullptr) { err = "vendor tier: null inner source"; return false; }
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { err = "vendor tier: socket() failed"; return false; }
    // Non-blocking connect with a 3 s deadline: a missing daemon must fail
    // fast, never hang startup.
    int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        err = "vendor tier: socket path too long";
        return false;
    }
    std::memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
    int rc = ::connect(fd, (sockaddr*) &addr, sizeof(addr));
    if (rc != 0 && errno != EINPROGRESS) {
        ::close(fd);
        err = "vendor tier: connect() refused (is the P40 daemon running?)";
        return false;
    }
    if (rc != 0) {
        fd_set wfds; FD_ZERO(&wfds); FD_SET(fd, &wfds);
        timeval tv{3, 0};
        if (::select(fd + 1, nullptr, &wfds, nullptr, &tv) <= 0) {
            ::close(fd);
            err = "vendor tier: connect() timed out";
            return false;
        }
        int soerr = 0; socklen_t sl = sizeof(soerr);
        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl);
        if (soerr != 0) {
            ::close(fd);
            err = "vendor tier: connect() failed (is the P40 daemon running?)";
            return false;
        }
    }
    ::fcntl(fd, F_SETFL, flags);
    timeval tv{10, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    std::vector<uint8_t> hello;
    put_u32(hello, kVersion);
    const char* name = "strata-hip-engine";
    put_u32(hello, (uint32_t) std::strlen(name));
    hello.insert(hello.end(), name, name + std::strlen(name));
    std::vector<uint8_t> frame;
    Header h{kMagic, HELLO, (uint64_t) hello.size()};
    frame.insert(frame.end(), (uint8_t*) &h, (uint8_t*) &h + sizeof(h));
    frame.insert(frame.end(), hello.begin(), hello.end());
    if (!send_all(fd, frame.data(), frame.size())) {
        ::close(fd);
        err = "vendor tier: HELLO send failed";
        return false;
    }
    Header rh{};
    if (!recv_all(fd, (uint8_t*) &rh, sizeof(rh)) || rh.magic != kMagic ||
        rh.op != HELLO_REPLY || rh.len > kMaxPayload || rh.len < 28) {
        ::close(fd);
        err = "vendor tier: bad HELLO_REPLY";
        return false;
    }
    std::vector<uint8_t> reply((size_t) rh.len);
    if (!recv_all(fd, reply.data(), reply.size())) {
        ::close(fd);
        err = "vendor tier: HELLO_REPLY read failed";
        return false;
    }
    uint32_t ver = 0; uint64_t bb = 0, nl = 0, ne = 0;
    std::memcpy(&ver, reply.data(), 4);
    std::memcpy(&bb, reply.data() + 4, 8);
    std::memcpy(&nl, reply.data() + 12, 8);
    std::memcpy(&ne, reply.data() + 20, 8);
    if (ver != kVersion || bb == 0) {
        ::close(fd);
        err = "vendor tier: version/blob mismatch";
        return false;
    }
    (void) layers; (void) experts; (void) nl; (void) ne;
    fd_ = fd;
    inner_ = inner;
    blob_bytes_ = bb;
    layers_ = (int64_t) nl;
    experts_ = (int64_t) ne;
    return true;
}

void VendorTierSource::close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ >= 0) {
        Header h{kMagic, BYE, 0};
        (void) send_all(fd_, (uint8_t*) &h, sizeof(h));
        ::close(fd_);
        fd_ = -1;
    }
    inner_ = nullptr;
}

const uint8_t* VendorTierSource::blob(int64_t layer, int64_t expert) {
    if (inner_ == nullptr) return nullptr;
    return inner_->blob(layer, expert);
}

bool VendorTierSource::fetch_one(int64_t layer, int64_t expert, uint8_t* dst) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ < 0 || inner_ == nullptr) return false;
    if (layer < 0 || expert < 0 || layer >= layers_ || expert >= experts_) return false;
    // Engine-side size for this pair (native packs vary per layer).
    uint64_t want = 0;
    try {
        want = strata::kernels::cpu::expert_layout().blob_bytes(layer);
    } catch (...) { return false; }
    if (want == 0 || want > kMaxPayload - 64) return false;
    uint8_t req[sizeof(Header) + 16];
    Header* h = (Header*) req;
    h->magic = kMagic; h->op = GET; h->len = 16;
    std::memcpy(req + sizeof(Header), &layer, 8);
    std::memcpy(req + sizeof(Header) + 8, &expert, 8);
    if (!send_all(fd_, req, sizeof(req))) return false;
    Header rh{};
    if (!recv_all(fd_, (uint8_t*) &rh, sizeof(rh))) return false;
    if (rh.magic != kMagic || rh.len > kMaxPayload) return false;
    if (rh.op == MISS) {
        __atomic_fetch_add(&tier_misses_, 1, __ATOMIC_RELAXED);
        return false;
    }
    if (rh.op != BLOB || rh.len < 48 || rh.len - 48 != want || rh.len - 48 != blob_bytes_) {
        // Size disagreement (different pack geometry): not ours to serve.
        // Drain the frame so the stream stays in sync, then fall back.
        std::vector<uint8_t> drain;
        drain.resize((size_t) (rh.len > kMaxPayload ? 0 : rh.len));
        if (!drain.empty() && !recv_all(fd_, drain.data(), drain.size())) return false;
        return false;
    }
    uint8_t meta[48];
    if (!recv_all(fd_, meta, sizeof(meta))) return false;
    int64_t rl = 0, re = 0;
    std::memcpy(&rl, meta, 8); std::memcpy(&re, meta + 8, 8);
    if (rl != layer || re != expert) return false;
    if (!recv_all(fd_, dst, (size_t) want)) return false;
    Sha256 s;
    s.update(dst, (size_t) want);
    auto digest = s.final();
    if (std::memcmp(digest.data(), meta + 16, 32) != 0) {
        __atomic_fetch_add(&tier_hash_mismatches_, 1, __ATOMIC_RELAXED);
        return false;
    }
    __atomic_fetch_add(&tier_hits_, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&reads_, 1, __ATOMIC_RELAXED);
    return true;
}

bool VendorTierSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (fetch_one(layer, expert, dst)) return true;
    if (inner_ == nullptr) return false;
    const auto& pl = strata::kernels::cpu::expert_layout();
    bool ok = inner_->copy_blob(layer, expert, dst);
    if (ok) __atomic_fetch_add(&tier_fallback_bytes_, (long long) pl.blob_bytes(layer), __ATOMIC_RELAXED);
    return ok;
}

bool VendorTierSource::copy_blobs(const int32_t* layers, const int32_t* experts,
                                  uint8_t* const* dst, size_t n) {
    // Per-pair GETs for phase 1 (correctness first; a FETCH_N batch op is
    // the measured follow-up once the hit path is proven). Falls back per
    // pair so one daemon MISS never fails the batch.
    bool all = true;
    for (size_t i = 0; i < n; ++i) {
        if (!copy_blob(layers[i], experts[i], dst[i])) all = false;
    }
    return all;
}

void VendorTierSource::prefetch(int64_t layer, const int64_t* experts, int64_t n) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (fd_ >= 0 && n > 0) {
    // Bounded batches of 64: hint the daemon, then hint inner_ too (it may
    // warm file pages for whatever the daemon misses).
    for (int64_t base = 0; base < n; base += 64) {
        int64_t m = n - base < 64 ? n - base : 64;
        std::vector<uint8_t> payload;
        put_u32(payload, (uint32_t) m);
        for (int64_t i = 0; i < m; ++i) {
            put_u64(payload, (uint64_t) layer);
            put_u64(payload, (uint64_t) experts[base + i]);
        }
        std::vector<uint8_t> frame;
        Header h{kMagic, PREFETCH, (uint64_t) payload.size()};
        frame.insert(frame.end(), (uint8_t*) &h, (uint8_t*) &h + sizeof(h));
        frame.insert(frame.end(), payload.begin(), payload.end());
        if (!send_all(fd_, frame.data(), frame.size())) break;
        Header rh{};
        if (!recv_all(fd_, (uint8_t*) &rh, sizeof(rh)) || rh.magic != kMagic) break;
        std::vector<uint8_t> rest;
        if (rh.len > 0 && rh.len <= kMaxPayload) {
            rest.resize((size_t) rh.len);
            if (!recv_all(fd_, rest.data(), rest.size())) break;
        }
        if (rh.op != ACK) break;
    }
        }
    }
    if (inner_ != nullptr) inner_->prefetch(layer, experts, n);
}

void VendorTierSource::warm(int64_t layer, const int64_t* experts, int64_t n) {
    prefetch(layer, experts, n);
    // FileExpertSource treats warm (page advice) and prefetch (reads)
    // separately: the tier hints reads via prefetch(), but the inner source
    // still needs its own page advice for whatever the daemon misses.
    if (inner_ != nullptr) inner_->warm(layer, experts, n);
}

int64_t VendorTierSource::reads() const {
    int64_t inner_reads = inner_ != nullptr ? inner_->reads() : 0;
    return reads_ + inner_reads;
}

}  // namespace strata::core

#else  // !__linux__

#include "strata/core/vendor_tier.hpp"
#include <string>
namespace strata::core {
VendorTierSource::~VendorTierSource() {}
bool VendorTierSource::open(const std::string&, ExpertSource*, int64_t, int64_t, std::string& err) {
    err = "vendor tier: Linux only";
    return false;
}
void VendorTierSource::close() {}
const uint8_t* VendorTierSource::blob(int64_t layer, int64_t expert) {
    return inner_ != nullptr ? inner_->blob(layer, expert) : nullptr;
}
bool VendorTierSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    return inner_ != nullptr && inner_->copy_blob(layer, expert, dst);
}
bool VendorTierSource::copy_blobs(const int32_t* l, const int32_t* e, uint8_t* const* d, size_t n) {
    return inner_ != nullptr && inner_->copy_blobs(l, e, d, n);
}
void VendorTierSource::prefetch(int64_t l, const int64_t* e, int64_t n) {
    if (inner_ != nullptr) inner_->prefetch(l, e, n);
}
void VendorTierSource::warm(int64_t l, const int64_t* e, int64_t n) {
    if (inner_ != nullptr) inner_->warm(l, e, n);
}
int64_t VendorTierSource::reads() const { return inner_ != nullptr ? inner_->reads() : 0; }
}  // namespace strata::core

#endif

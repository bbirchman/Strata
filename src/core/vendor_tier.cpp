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
// Shared protocol constants (framing ops live locally below; these asserts
// stop the two copies drifting silently - the daemon includes page_wire.hpp
// as its single source of truth).
#include "strata/core/page_wire.hpp"
static_assert(strata::core::pagewire::kMagic == 0x50343054, "page protocol magic");
static_assert(strata::core::pagewire::kVersion == 1, "page protocol version");
static_assert(strata::core::pagewire::kMaxPayload == (8ull << 20), "page max payload");
static_assert((uint32_t) strata::core::pagewire::Op::BYE == 10, "page op numbering");
// cudaStream_t for the note_async_read stream queries below. Every other
// core TU already includes this (it resolves to HIP in HIP builds).
#include <cuda_runtime.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
                     PREFETCH = 6, ACK = 7, STATS = 8, STATS_REPLY = 9, BYE = 10, ERROR = 11,
                     FETCH_N = 12 };

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
// STRATA_P40_TRACE=1: log the first 500 socket ops (op, layer, expert) to
// stderr so a flood names its caller. Off by default (hot path).
bool trace_on() {
    static bool v = [] {
        const char* e = std::getenv("STRATA_P40_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}
long long trace_next() {
    static long long n = 0;
    return __atomic_fetch_add(&n, 1, __ATOMIC_RELAXED);
}
constexpr long long kTraceCap = 500;
// Decision tracing (STRATA_P40_TRACE=1, first 20): which branch each blob()
// takes. Wire tracing above only fires when a GET is sent; silence there
// cannot tell "never called" from "always delegated".
void trace_decision(const char* what, int64_t layer, int64_t expert, const char* why) {
    if (trace_on() && trace_next() < 20)
        std::fprintf(stderr, "p40tier: %s (%lld,%lld) -> %s\n", what, (long long) layer,
                     (long long) expert, why);
}
// White-box timing: RAII microsecond accumulator (anonymous-namespace,
// header-free). Put one past the eligibility checks in each fetch path;
// per-request tier time = delta of tier_us().
struct TierTimer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    int64_t* dst = nullptr;
    explicit TierTimer(int64_t* d) : dst(d) {}
    ~TierTimer() {
        __atomic_fetch_add(dst,
                           (long long) std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - t0)
                               .count(),
                           __ATOMIC_RELAXED);
    }
};
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
    inner_ = inner;
    return connect_daemon(socket_path, layers, experts, err);
}

bool VendorTierSource::open_tier(const std::string& socket_path, int64_t layers, int64_t experts,
                                 std::string& err) {
    // Self mode: the base part is already open as the file tier (generate.cpp
    // opened it); only the daemon side is (re)connected here. close() only
    // resets tier state (inner_ is already null), never the file mapping.
    close();
    return connect_daemon(socket_path, layers, experts, err);
}

bool VendorTierSource::connect_daemon(const std::string& socket_path, int64_t layers, int64_t experts,
                                      std::string& err, bool sentinel_proof) {
    // Remember for lazy reconnect after a mid-stream failure (drop_tier):
    // a transient blip must not kill the tier for the session.
    socket_path_ = socket_path;
    cfg_layers_ = layers;
    cfg_experts_ = experts;
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
    // Bulk BLOBs (2+ MB) must never wedge against a peer that recvs normally:
    // 16 MB buffers absorb a whole frame either way.
    int bigbuf = 16 << 20;
    ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bigbuf, sizeof(bigbuf));
    ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bigbuf, sizeof(bigbuf));

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
    (void) layers;
    (void) experts;
    (void) nl;
    (void) ne;
    // HELLO_REPLY field 5 (optional, v1.1): the daemon's VRAM-cached bytes.
    // Absent (older/file-mode daemons, Python fixture) reads 0 and blob()
    // delegates to the inner source.
    uint64_t vc = 0;
    if (reply.size() >= 36) std::memcpy(&vc, reply.data() + 28, 8);
    // HELLO_REPLY field 6 (optional, v1.2): the daemon's held set, one bit
    // per (layer, expert) in row-major order over the DAEMON's geometry.
    // Exact-size match only; anything else means "unknown", i.e. ask.
    std::vector<uint8_t> held;
    if (nl > 0 && ne > 0 && (uint64_t) nl <= 1024 && (uint64_t) ne <= 4096) {
        uint64_t mapbytes = ((uint64_t) nl * (uint64_t) ne + 7) / 8;
        if (reply.size() == 36 + (size_t) mapbytes)
            held.assign(reply.begin() + 36, reply.end());
    }
    // Always reported (one line per run, startup-only): geometry as parsed,
    // the daemon's VRAM cache, and the held-bitmap state behind blob().
    std::fprintf(stderr,
                 "p40tier: HELLO geometry %llu x %llu, daemon VRAM cache %.2f GiB, bitmap %zu bytes (%s)\n",
                 (unsigned long long) nl, (unsigned long long) ne, (double) vc / 1073741824.0,
                 held.size(), held.empty() ? "absent: asking daemon for everything" : "parsed");
    fd_ = fd;
    blob_bytes_ = bb;
    layers_ = (int64_t) nl;
    experts_ = (int64_t) ne;
    vram_cached_ = vc;
    held_bits_ = std::move(held);
    tier_skipped_ = 0;
    staging_.clear();
    staging_bytes_ = 0;
    staging_next_ = 0;
    if (vc > 0) {
        // Native packs vary per layer (up to ~2.7 MB observed) while
        // max_blob can still read the canonical default: take the true max
        // over every layer, else every blob() call fails the size gate and
        // the tier silently never serves (XXS mmap symptom: 0 GETs).
        uint64_t mb = 0;
        try {
            const auto& pl = strata::kernels::cpu::expert_layout();
            for (int64_t l = 0; l < nl; ++l) {
                uint64_t b = pl.blob_bytes(l);
                if (b > mb) mb = b;
            }
        } catch (...) { mb = 0; }
        if (mb > 0 && mb <= kMaxPayload) {
            staging_.resize(kStagingSlots);
            for (auto& s : staging_) {
                s.ptr = (uint8_t*) std::malloc((size_t) mb);
                if (s.ptr == nullptr) {
                    for (auto& t : staging_) std::free(t.ptr);
                    staging_.clear();
                    break;
                }
            }
            if (!staging_.empty()) staging_bytes_ = mb;
        }
    }
    // Sentinel verification (fail-open): the first 3 daemon-
    // held pairs are fetched and compared byte-for-byte against the inner
    // source (the pack). This is the content proof that replaces per-blob
    // hashing: same guarantee for pack identity and transfer integrity at
    // ~6 blobs of one-time cost instead of ~8 ms per hit forever.
    // Skipped on reconnect (caller holds mu_; proof from first connect stands).
    if (sentinel_proof && !staging_.empty() && !held.empty()) {
        int checked = 0, matched = 0;
        for (int64_t l = 0; l < nl && checked < 3; ++l) {
            for (int64_t e = 0; e < ne && checked < 3; ++e) {
                uint64_t i = (uint64_t) l * (uint64_t) ne + (uint64_t) e;
                if ((held[(size_t) (i >> 3)] & (uint8_t) (1u << (i & 7))) == 0) continue;
                uint64_t want = 0;
                try {
                    want = strata::kernels::cpu::expert_layout().blob_bytes(l);
                } catch (...) { continue; }
                if (want == 0 || want > staging_bytes_) continue;
                std::vector<uint8_t> a((size_t) want), b((size_t) want);
                if (!fetch_one(l, (int32_t) e, a.data())) continue;
                if (!fb_copy_blob(l, (int32_t) e, b.data())) continue;
                ++checked;
                if (std::memcmp(a.data(), b.data(), (size_t) want) == 0) ++matched;
            }
        }
        std::fprintf(stderr, "p40tier: sentinel verify: %d/%d daemon blobs byte-identical with the pack%s\n",
                     matched, checked, (checked > 0 && matched < checked) ? " (WARNING: continuing anyway)" : "");
    }
    return true;
}

void VendorTierSource::close() {
    std::lock_guard<std::mutex> lock(mu_);
    close_locked();
    inner_ = nullptr;
    socket_path_.clear();
}

void VendorTierSource::close_locked() {
    if (fd_ >= 0) {
        Header h{kMagic, BYE, 0};
        (void) send_all(fd_, (uint8_t*) &h, sizeof(h));
        ::close(fd_);
        fd_ = -1;
    }
    for (auto& s : staging_) std::free(s.ptr);
    staging_.clear();
    staging_bytes_ = 0;
    staging_next_ = 0;
    vram_cached_ = 0;
    held_bits_.clear();
    inner_ = nullptr;
}

// Drop a poisoned stream WITHOUT BYE (the peer may be wedged; even BYE
// could block): kernel close delivers FIN/RST so the daemon's next
// send/recv fails fast and it returns to accept(). Keeps inner_ and the
// base mapping: the tier degrades to file until try_reconnect_locked().
void VendorTierSource::drop_tier_locked(const char* why) {
    // errno FIRST: fprintf/free/clock clobber it. On logic failures (addr
    // mismatch) it just reports a stale value (= "no transport error").
    int e = errno;
    char ebuf[80];
    (void) strerror_r(e, ebuf, sizeof(ebuf));
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    for (auto& s : staging_) std::free(s.ptr);
    staging_.clear();
    staging_bytes_ = 0;
    staging_next_ = 0;
    vram_cached_ = 0;
    held_bits_.clear();
    ++tier_drops_;
    std::fprintf(stderr, "p40tier: transport failure (%s, errno %d=%s): dropped the daemon connection (fail-open to file)\n",
                     why ? why : "?", e, ebuf);
}

// Re-dial after a drop (throttled: at most one attempt per 30 s so a dead
// daemon costs one 3 s connect per half minute, never per miss).
bool VendorTierSource::try_reconnect_locked() {
    if (fd_ >= 0 || socket_path_.empty()) return fd_ >= 0;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_reconnect_ < std::chrono::seconds(30)) return false;
    last_reconnect_ = now;
    std::string err;
    // close_locked() first (no BYE to a dead peer... BYE to a live one is
    // polite but close_locked sends it; fd_ is -1 here so it only resets
    // staging state): then a fresh HELLO + bitmap + staging + sentinels.
    close_locked();
    if (connect_daemon(socket_path_, cfg_layers_, cfg_experts_, err, /*sentinel_proof=*/false)) {
        ++tier_reconnects_;
        std::fprintf(stderr, "p40tier: reconnected to the daemon\n");
        return true;
    }
    std::fprintf(stderr, "p40tier: reconnect failed (%s)\n", err.c_str());
    return false;
}

const uint8_t* VendorTierSource::blob(int64_t layer, int64_t expert) {
    std::lock_guard<std::mutex> lock(mu_);
    // File-backed daemon (or no staging): the inner source is already the
    // fastest path, so delegate. The daemon-first path below only pays when
    // the daemon holds a VRAM cache the host does not have.
    // Dropped connection: try one throttled re-dial (a transient blip
    // must not park the tier on file for the rest of the session).
    if (fd_ < 0) {
        if (!try_reconnect_locked()) {
            trace_decision("blob", layer, expert, "delegate: no daemon");
            return fb_blob(layer, expert);
        }
    }
    if (vram_cached_ == 0 || staging_.empty() || staging_bytes_ == 0) {
        trace_decision("blob", layer, expert, "delegate: file-mode/no-staging");
        return fb_blob(layer, expert);
    }
    if (layer < 0 || expert < 0 || layer >= layers_ || expert >= experts_) {
        trace_decision("blob", layer, expert, "delegate: out of range");
        return fb_blob(layer, expert);
    }
    uint64_t want = 0;
    try {
        want = strata::kernels::cpu::expert_layout().blob_bytes(layer);
    } catch (...) { return fb_blob(layer, expert); }
    if (want == 0 || want > staging_bytes_) {
        trace_decision("blob", layer, expert, "delegate: bad size");
        return fb_blob(layer, expert);
    }
    if (!daemon_holds(layer, expert)) {
        __atomic_fetch_add(&tier_skipped_, 1, __ATOMIC_RELAXED);
        trace_decision("blob", layer, expert, "delegate: bitset-unheld");
        return fb_blob(layer, expert);
    }
    trace_decision("blob", layer, expert, "daemon-first");
    for (size_t n = 0; n < staging_.size(); ++n) {
        size_t i = (staging_next_ + n) % staging_.size();
        StagingSlot& s = staging_[i];
        if (s.busy) {
            // Reuse only when the stream reports done; anything else
            // (not-ready or an error) keeps the slot busy. No blocking.
            if (s.stream == nullptr) continue;
            if (::cudaStreamQuery((cudaStream_t) s.stream) != cudaSuccess) continue;
            (void) ::cudaGetLastError();  // clear a sticky query error, if any
            s.busy = false;
            s.stream = nullptr;
        }
        if (fetch_one_locked(layer, expert, s.ptr)) {
            staging_next_ = (i + 1) % staging_.size();
            return s.ptr;
        }
        return fb_blob(layer, expert);
    }
    return fb_blob(layer, expert);
}

void VendorTierSource::note_async_read(const uint8_t* blob, void* stream) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& s : staging_) {
        if (s.ptr != nullptr && blob >= s.ptr && blob < s.ptr + staging_bytes_) {
            s.busy = true;
            s.stream = stream;
            return;
        }
    }
    fb_note(blob, stream);
}

bool VendorTierSource::fetch_one(int64_t layer, int64_t expert, uint8_t* dst) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return fetch_one_locked(layer, expert, dst);
}

bool VendorTierSource::fetch_one_locked(int64_t layer, int64_t expert, uint8_t* dst) noexcept {
    if (fd_ < 0) return false;
    if (!daemon_holds(layer, expert)) {
        __atomic_fetch_add(&tier_skipped_, 1, __ATOMIC_RELAXED);
        return false;
    }
    if (layer < 0 || expert < 0 || layer >= layers_ || expert >= experts_) return false;
    // Engine-side size for this pair (native packs vary per layer).
    uint64_t want = 0;
    try {
        want = strata::kernels::cpu::expert_layout().blob_bytes(layer);
    } catch (...) { return false; }
    if (want == 0 || want > kMaxPayload - 64) return false;
    TierTimer _tt(&tier_us_);
    uint8_t req[sizeof(Header) + 16];
    Header* h = (Header*) req;
    h->magic = kMagic; h->op = GET; h->len = 16;
    std::memcpy(req + sizeof(Header), &layer, 8);
    std::memcpy(req + sizeof(Header) + 8, &expert, 8);
    const bool timed = trace_on();
    const auto t0 = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (!send_all(fd_, req, sizeof(req))) {
        if (trace_on() && trace_next() < kTraceCap)
            std::fprintf(stderr, "p40tier: GET (%lld,%lld) send-fail\n", (long long) layer,
                         (long long) expert);
        drop_tier_locked("one:send");
        return false;
    }
    if (trace_on() && trace_next() < kTraceCap)
        std::fprintf(stderr, "p40tier: GET (%lld,%lld) sent\n", (long long) layer, (long long) expert);
    Header rh{};
    if (!recv_all(fd_, (uint8_t*) &rh, sizeof(rh))) {
        drop_tier_locked("one:recv-hdr");
        return false;
    }
    const auto t1 = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (rh.magic != kMagic || rh.len > kMaxPayload) {
        drop_tier_locked("one:bad-hdr");
        return false;
    }
    if (rh.op == MISS) {
        __atomic_fetch_add(&tier_misses_, 1, __ATOMIC_RELAXED);
        if (trace_on() && trace_next() < kTraceCap)
            std::fprintf(stderr, "p40tier: GET (%lld,%lld) -> MISS\n", (long long) layer,
                         (long long) expert);
        return false;
    }
    if (rh.op != BLOB || rh.len < 48 || rh.len - 48 != want) {
        // Size disagreement (daemon serves a different pack geometry):
        // not ours to serve. Drain the frame so the stream stays in sync,
        // then fall back. (Pack identity is the operator's job: the daemon
        // must open the same --pack as the engine.)
        std::vector<uint8_t> drain;
        drain.resize((size_t) (rh.len > kMaxPayload ? 0 : rh.len));
        if (!drain.empty() && !recv_all(fd_, drain.data(), drain.size())) {
            drop_tier_locked("one:drain");
            return false;
        }
        return false;
    }
    uint8_t meta[48];
    if (!recv_all(fd_, meta, sizeof(meta))) {
        drop_tier_locked("one:meta");
        return false;
    }
    int64_t rl = 0, re = 0;
    std::memcpy(&rl, meta, 8); std::memcpy(&re, meta + 8, 8);
    // Address mismatch: the daemon answered for another pair; its blob body
    // is still on the wire -> the stream is desynced. Drop, never continue.
    if (rl != layer || re != expert) {
        drop_tier_locked("one:addr");
        return false;
    }
    if (!recv_all(fd_, dst, (size_t) want)) {
        drop_tier_locked("one:blob");
        return false;
    }
    if (timed) {
        const auto t2 = std::chrono::steady_clock::now();
        const long long first_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        const long long rest_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        static long long n = 0;
        if (__atomic_fetch_add(&n, 1, __ATOMIC_RELAXED) < 30)
            std::fprintf(stderr, "p40tier: perf (%lld,%lld) %llu B: req+reply-header %lld us, meta+blob %lld us\n",
                         (long long) layer, (long long) expert, (unsigned long long) want,
                         first_us, rest_us);
    }
    // Integrity without per-blob hashing: size + address are checked on
    // every blob (framing bugs fail closed); content is proven once per
    // session by sentinel GETs at open() and by the daemon's fill-time
    // reads. STRATA_P40_VERIFY=1 restores per-blob sha256 (both ends must
    // set it: the daemon sends zeros otherwise) for A/B measurements.
    // A 2 MB sha256 pair costs ~8 ms of CPU per hit - more than the PCIe
    // transfer it protects (10,799 hits averaged ~1 tok/s with it on).
    static bool verify = [] {
        const char* e = std::getenv("STRATA_P40_VERIFY");
        return e != nullptr && e[0] == '1';
    }();
    if (verify) {
        Sha256 s;
        s.update(dst, (size_t) want);
        auto digest = s.final();
        if (std::memcmp(digest.data(), meta + 16, 32) != 0) {
            __atomic_fetch_add(&tier_hash_mismatches_, 1, __ATOMIC_RELAXED);
            return false;
        }
    }
    __atomic_fetch_add(&tier_hits_, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&reads_, 1, __ATOMIC_RELAXED);
    if (trace_on() && trace_next() < kTraceCap)
        std::fprintf(stderr, "p40tier: GET (%lld,%lld) -> BLOB %llu B\n", (long long) layer,
                     (long long) expert, (unsigned long long) want);
    return true;
}

bool VendorTierSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (fetch_one(layer, expert, dst)) return true;
    // One throttled re-dial, then one retry: a dropped stream recovers
    // inside this call instead of parking on file for 30 s.
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (try_reconnect_locked() && fetch_one_locked(layer, expert, dst)) return true;
    }
    const auto& pl = strata::kernels::cpu::expert_layout();
    bool ok = fb_copy_blob(layer, expert, dst);
    if (ok) __atomic_fetch_add(&tier_fallback_bytes_, (long long) pl.blob_bytes(layer), __ATOMIC_RELAXED);
    return ok;
}

// One FETCH_N round trip for the eligible pairs. ok[i] = blob received and
// accepted. Returns false only when the stream broke (caller falls back) or
// the daemon predates FETCH_N (ERROR on the first reply: caller retries the
// whole set per-pair). mu_ already held (copy_blobs holds it; the socket is
// one stream: frames must never interleave).
bool VendorTierSource::fetch_batch_locked(const int32_t* layers, const int32_t* experts,
                                          uint8_t* const* dst, const uint64_t* want,
                                          const size_t* idx, size_t m, char* ok) noexcept {
    if (fd_ < 0 || m == 0 || m > 64) return false;
    for (size_t k = 0; k < m; ++k) ok[idx[k]] = false;
    TierTimer _tt(&tier_us_);
    __atomic_fetch_add(&tier_batches_, 1, __ATOMIC_RELAXED);
    std::vector<uint8_t> req;
    req.reserve(sizeof(Header) + 4 + m * 16);
    Header h{kMagic, FETCH_N, (uint64_t)(4 + m * 16)};
    req.insert(req.end(), (uint8_t*) &h, (uint8_t*) &h + sizeof(h));
    uint32_t mn = (uint32_t) m;
    req.insert(req.end(), (uint8_t*) &mn, (uint8_t*) &mn + 4);
    for (size_t k = 0; k < m; ++k) {
        int64_t l = layers[idx[k]], e = experts[idx[k]];
        req.insert(req.end(), (uint8_t*) &l, (uint8_t*) &l + 8);
        req.insert(req.end(), (uint8_t*) &e, (uint8_t*) &e + 8);
    }
    if (!send_all(fd_, req.data(), req.size())) {
        drop_tier_locked("batch:send");
        return false;
    }
    static bool verify = [] {
        const char* e = std::getenv("STRATA_P40_VERIFY");
        return e != nullptr && e[0] == '1';
    }();
    for (size_t k = 0; k < m; ++k) {
        Header rh{};
        if (!recv_all(fd_, (uint8_t*) &rh, sizeof(rh))) {
            drop_tier_locked("batch:recv-hdr");
            return false;
        }
        if (rh.magic != kMagic || rh.len > kMaxPayload) {
            drop_tier_locked("batch:bad-hdr");
            return false;
        }
        if (rh.op == ERROR) {
            // Old daemon (pre-FETCH_N): exactly one ERROR frame stands in
            // for the whole batch. Report unsupported: the caller retries
            // per-pair GETs, which every daemon speaks. Stream stays synced
            // (the single frame is consumed); no drop unless the drain fails.
            for (size_t j = k; j < m; ++j) ok[idx[j]] = false;
            if (rh.len > 0) {
                std::vector<uint8_t> drain((size_t) rh.len);
                if (!recv_all(fd_, drain.data(), drain.size())) {
                    drop_tier_locked("batch:err-drain");
                    return false;
                }
            }
            return k == 0 ? false : true;  // false = retry whole batch per-pair
        }
        if (rh.op == MISS) {
            if (rh.len != 16) {
                drop_tier_locked("batch:miss-len");
                return false;
            }
            uint8_t mm[16];
            if (!recv_all(fd_, mm, sizeof(mm))) {
                drop_tier_locked("batch:miss-body");
                return false;
            }
            __atomic_fetch_add(&tier_misses_, 1, __ATOMIC_RELAXED);
            ok[idx[k]] = false;
            continue;
        }
        if (rh.op != BLOB || rh.len < 48 || rh.len - 48 != want[idx[k]]) {
            std::vector<uint8_t> drain((size_t) (rh.len > kMaxPayload ? 0 : rh.len));
            if (!drain.empty() && !recv_all(fd_, drain.data(), drain.size())) {
                drop_tier_locked("batch:drain");
                return false;
            }
            ok[idx[k]] = false;
            continue;
        }
        uint8_t meta[48];
        if (!recv_all(fd_, meta, sizeof(meta))) {
            drop_tier_locked("batch:meta");
            return false;
        }
        int64_t rl = 0, re = 0;
        std::memcpy(&rl, meta, 8); std::memcpy(&re, meta + 8, 8);
        if (rl != (int64_t) layers[idx[k]] || re != (int64_t) experts[idx[k]]) {
            drop_tier_locked("batch:addr");
            return false;
        }
        if (!recv_all(fd_, dst[idx[k]], (size_t) want[idx[k]])) {
            drop_tier_locked("batch:blob");
            return false;
        }
        if (verify) {
            Sha256 s;
            s.update(dst[idx[k]], (size_t) want[idx[k]]);
            auto digest = s.final();
            if (std::memcmp(digest.data(), meta + 16, 32) != 0) {
                __atomic_fetch_add(&tier_hash_mismatches_, 1, __ATOMIC_RELAXED);
                ok[idx[k]] = false;
                continue;
            }
        }
        __atomic_fetch_add(&tier_hits_, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&reads_, 1, __ATOMIC_RELAXED);
        ok[idx[k]] = true;
    }
    if (trace_on() && trace_next() < kTraceCap) {
        long long good = 0;
        for (size_t k = 0; k < m; ++k) good += ok[idx[k]] ? 1 : 0;
        std::fprintf(stderr, "p40tier: FETCH_N %zu pairs -> %lld blobs\n", m, good);
    }
    return true;
}

bool VendorTierSource::copy_blobs(const int32_t* layers, const int32_t* experts,
                                  uint8_t* const* dst, size_t n) {
    // Batched FETCH_N (one RTT per 64 pairs) with per-pair fallback, so one
    // daemon MISS never fails the batch. Pairs the daemon cannot hold go
    // straight to the inner source with zero socket cost.
    if (n == 0) return true;
    const auto& pl = strata::kernels::cpu::expert_layout();
    std::vector<uint64_t> want(n, 0);
    std::vector<size_t> idx;
    idx.reserve(n);
    std::vector<char> batched(n, 0);
    std::vector<char> ok(n, 0); // char: no vector<bool> bitpacking
    {
        // One critical section for partition + all batch round trips: the
        // socket is one stream and held_bits_/fd_ are shared. The file
        // fallback below runs OUTSIDE the lock (the base copy re-enters
        // tier methods: holding mu_ across it deadlocked the engine).
        std::lock_guard<std::mutex> lock(mu_);
        // Dropped connection: one throttled re-dial before partitioning.
        if (fd_ < 0) (void) try_reconnect_locked();
        if (fd_ >= 0) {
            for (size_t i = 0; i < n; ++i) {
                int64_t l = layers[i], e = experts[i];
                if (l < 0 || e < 0 || l >= layers_ || e >= experts_) continue;
                if (!daemon_holds(l, e)) {
                    __atomic_fetch_add(&tier_skipped_, 1, __ATOMIC_RELAXED);
                    continue;
                }
                uint64_t w = 0;
                try { w = pl.blob_bytes(l); } catch (...) { continue; }
                if (w == 0 || w > kMaxPayload - 64) continue;
                want[i] = w;
                idx.push_back(i);
            }
            for (size_t c = 0; c < idx.size(); c += 64) {
                size_t m = std::min<size_t>(64, idx.size() - c);
                // fetch_batch_locked marks ok[]; false = transport broke
                // (dropped inside: remaining pairs fall back) or an old
                // daemon (retry per-pair GETs below via the same fallback).
                if (!fetch_batch_locked(layers, experts, dst, want.data(), idx.data() + c, m,
                                        ok.data())) {
                    // Transport broke mid-batch: drop already done inside;
                    // fall back per-pair below (GETs re-dial throttled).
                    for (size_t k = 0; k < m; ++k)
                        if (!ok[idx[c + k]]) batched[idx[c + k]] = 0;
                }
                for (size_t k = 0; k < m; ++k) batched[idx[c + k]] = ok[idx[c + k]];
            }
        }
    }
    bool all = true;
    for (size_t i = 0; i < n; ++i) {
        if (batched[i]) continue;
        // Per-pair GET (speaks to old daemons too) then file. Bytes counted
        // inside copy_blob; MISS/second-failure just falls back again.
        if (!copy_blob(layers[i], experts[i], dst[i])) all = false;
    }
    return all;
}

void VendorTierSource::prefetch(int64_t layer, const int64_t* experts, int64_t n) {
    // The daemon needs no warming: it either holds the pair (served on
    // demand) or not (MISS). Forward to inner_ only, so the hot path pays
    // zero socket round trips here (a PREFETCH+ACK per layer per token
    // measured as pure overhead: 9.8 -> 6.4 tok/s with zero GETs served).
    fb_prefetch(layer, experts, n);
}

void VendorTierSource::warm(int64_t layer, const int64_t* experts, int64_t n) {
    prefetch(layer, experts, n);
    // FileExpertSource treats warm (page advice) and prefetch (reads)
    // separately: the tier hints reads via prefetch(), but the inner source
    // still needs its own page advice for whatever the daemon misses.
    fb_warm(layer, experts, n);
}

int64_t VendorTierSource::reads() const {
    return reads_ + fb_reads();
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

// include/strata/core/vendor_tier.hpp - het-p40-tier: a P40 VRAM expert tier
// behind a Unix socket, as an ExpertSource decorator.
//
// A VendorTierSource wraps another source (the file tier or the arena) and
// serves copy_blob/copy_blobs from a CUDA daemon holding blobs in P40 VRAM.
// Anything the daemon cannot serve - a MISS, a socket error, a hash mismatch
// - falls back to the wrapped source, so the tier is advisory: with the flag
// off nothing changes (byte-identical default), with it on a failure only
// costs speed, never correctness.
//
// Backend-neutral on purpose: no cuda_runtime.h here (unlike
// remote_experts.hpp, which is CUDA-only). The HIP engine links this file
// unchanged; the CUDA daemon is a separate process and build.
#pragma once

#include "strata/core/expert_source.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

// Protocol v1, tools/p40_tier/page_proto.py: frames are
// [u32 magic 0x50343054][u32 op][u64 len][payload], little-endian.
// Ops: HELLO=1, HELLO_REPLY=2, GET=3, BLOB=4, MISS=5, PREFETCH=6, ACK=7,
// STATS=8, STATS_REPLY=9, BYE=10, ERROR=11.
class VendorTierSource : public ExpertSource {
public:
    VendorTierSource() = default;
    ~VendorTierSource() override;
    VendorTierSource(const VendorTierSource&) = delete;
    VendorTierSource& operator=(const VendorTierSource&) = delete;

    /// Connect to the daemon at `socket_path` and wrap `inner` (not owned).
    /// False with `err` when the daemon cannot be reached: the caller keeps
    /// the inner source and runs without the tier.
    bool open(const std::string& socket_path, ExpertSource* inner,
              int64_t layers, int64_t experts, std::string& err);
    void close();
    bool valid() const { return fd_ >= 0; }
    /// The wrapped source (null when closed): mode checks compare against
    /// this, so a wrapped arena still counts as the arena.
    ExpertSource* inner() const { return inner_; }
    /// Whether the daemon holds a VRAM cache (HELLO_REPLY field 5 > 0):
    /// only then does blob() fetch daemon-first (a file-backed daemon is
    /// no faster than the inner source, so blob() delegates to it).
    bool vram_backed() const { return vram_cached_ > 0; }

    // ExpertSource: blob() fetches daemon-first (VRAM-mode daemons only;
    // file-mode delegates), copy_blob(s) try the daemon then fall back per
    // pair, prefetch/warm hint both daemon and inner.
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool copy_blobs(const int32_t* layers, const int32_t* experts,
                    uint8_t* const* dst, size_t n) override;
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override;
    void warm(int64_t layer, const int64_t* experts, int64_t n) override;
    bool warms() const override { return valid(); }
    int64_t reads() const override;

    // Counters, for the driver to report rather than for control flow.
    int64_t tier_hits() const { return tier_hits_; }
    int64_t tier_misses() const { return tier_misses_; }
    int64_t tier_fallback_bytes() const { return tier_fallback_bytes_; }
    int64_t tier_hash_mismatches() const { return tier_hash_mismatches_; }

private:
    // One GET round trip. Returns true with bytes in dst on BLOB with a
    // matching sha256; false on MISS or any transport error (caller falls
    // back to inner_). Never throws.
    bool fetch_one(int64_t layer, int64_t expert, uint8_t* dst) noexcept;
    // fetch_one with mu_ already held (blob() holds it across slot scan +
    // fetch so two threads can never share a staging slot).
    bool fetch_one_locked(int64_t layer, int64_t expert, uint8_t* dst) noexcept;

    int fd_ = -1;
    ExpertSource* inner_ = nullptr;
    // blob() daemon path: rotating staging slots (the contract keeps every
    // handed-out pointer valid, and the pool can hold several while it
    // computes). note_async_read() marks a slot busy; a slot is reused only
    // when free or its stream reports done, else the call falls back to
    // inner_ (never blocks, never recycles a live read).
    struct StagingSlot {
        uint8_t* ptr = nullptr;
        bool busy = false;
        void* stream = nullptr;
    };
    static constexpr size_t kStagingSlots = 64;
    std::vector<StagingSlot> staging_;
    uint64_t staging_bytes_ = 0;  ///< per-slot bytes (layout max_blob)
    size_t staging_next_ = 0;
    uint64_t vram_cached_ = 0;  ///< daemon VRAM bytes (HELLO_REPLY field 5)
    // The socket is one stream: all send/recv sequences take mu_ so two
    // calling threads can never interleave frames (copy_blobs runs on the
    // prompt stager's threads).
    mutable std::mutex mu_;
    uint64_t blob_bytes_ = 0;
    int64_t layers_ = 0, experts_ = 0;
    int64_t tier_hits_ = 0, tier_misses_ = 0;
    int64_t tier_fallback_bytes_ = 0, tier_hash_mismatches_ = 0;
    int64_t reads_ = 0;
};

}  // namespace strata::core

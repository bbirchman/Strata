// include/strata/core/vendor_tier.hpp - het-p40-tier: a P40 VRAM expert tier
// behind a Unix socket, as a FileExpertSource SUBCLASS.
//
// Why subclass, not wrapper: generate.cpp drives the concrete FileExpertSource
// object for all bulk movement (resident copies, prefill staging, mapped
// reads); a pure ExpertSource wrapper only sees the abstract srcp paths the
// file modes barely use (0 GETs across whole XXS sessions proved it). As a
// subclass substituted for the single `src` local, every existing call site
// (concrete methods, &src comparisons, FileExpertSource& params) keeps
// working untouched: flag off, every override delegates to the base and the
// default path is byte-identical by construction.
//
// Two modes: (a) self mode (file/mmap/budget): inner_ is null and fallbacks
// call the base implementation - the object IS the file tier plus a daemon;
// (b) arena-wrap mode: set_inner(&arena_src) and fallbacks call inner_.
// Backend-neutral: no cuda_runtime.h here. The HIP engine links this file
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
class VendorTierSource : public FileExpertSource {
public:
    VendorTierSource() = default;
    ~VendorTierSource() override;
    VendorTierSource(const VendorTierSource&) = delete;
    VendorTierSource& operator=(const VendorTierSource&) = delete;

    /// Arena-wrap mode: connect to the daemon at `socket_path` and fall back
    /// to `inner` (not owned). False with `err` when the daemon cannot be
    /// reached: the caller keeps the selected source and runs tierless.
    bool open(const std::string& socket_path, ExpertSource* inner,
              int64_t layers, int64_t experts, std::string& err);
    /// Self mode (file/mmap/budget): the base part is already open as the
    /// file tier; connect the daemon for it. Same fail-open contract.
    bool open_tier(const std::string& socket_path, int64_t layers, int64_t experts,
                   std::string& err);
    /// Arena-wrap target (not owned). Null (default) = self mode.
    void set_inner(ExpertSource* inner) { inner_ = inner; }
    void close();
    bool valid() const { return fd_ >= 0; }
    /// The wrapped source (null when closed): mode checks compare against
    /// this, so a wrapped arena still counts as the arena.
    ExpertSource* inner() const { return inner_; }
    /// The daemon's held set as a bitset over (layer, expert) (HELLO_REPLY
    /// field 6, empty = unknown: ask the daemon). Checking it first keeps
    /// hot pairs the engine already holds off the socket entirely.
    bool daemon_holds(int64_t layer, int64_t expert) const {
        if (held_bits_.empty() || layer < 0 || expert < 0 || layer >= layers_ || expert >= experts_)
            return true;  // unknown: ask (fail-open direction)
        uint64_t i = (uint64_t) layer * (uint64_t) experts_ + (uint64_t) expert;
        return (held_bits_[i >> 3] & (uint8_t) (1u << (i & 7))) != 0;
    }
    /// Whether the daemon holds a VRAM cache (HELLO_REPLY field 5 > 0):
    /// only then does blob() fetch daemon-first (a file-backed daemon is
    /// no faster than the inner source, so blob() delegates to it).
    bool vram_backed() const { return vram_cached_ > 0; }

    // ExpertSource: blob() fetches daemon-first (VRAM-mode daemons only;
    // file-mode delegates), copy_blob(s) try the daemon then fall back per
    // pair, prefetch/warm hint both daemon and inner.
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    // Prefill fills and swap-ins read through blob_stable (the mapping-aware
    // path), not blob: without this override the tier watches a road nothing
    // drives (0 GETs across whole XXS sessions proved it). Same daemon-first
    // logic as blob(); staging slots are stream-guarded by note_async_read,
    // so a recycled slot is never handed out while a DMA still reads it.
    // Callers that copy immediately (fills, stagers) are unaffected by
    // recycling; long holders are covered by the busy guard.
    const uint8_t* blob_stable(int64_t layer, int64_t expert) override { return blob(layer, expert); }
    void note_async_read(const uint8_t* blob, void* stream) override;
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
    int64_t tier_skipped() const { return tier_skipped_; }
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
    // Shared dial+HELLO+staging body behind open() and open_tier().
    bool connect_daemon(const std::string& socket_path, int64_t layers, int64_t experts,
                        std::string& err);

    int fd_ = -1;
    ExpertSource* inner_ = nullptr;  ///< arena-wrap target; null = self mode
    // Fallbacks: arena-wrap mode calls inner_, self mode calls the base
    // implementation (qualified calls: no virtual redispatch, no recursion).
    const uint8_t* fb_blob(int64_t l, int64_t e) {
        return inner_ != nullptr ? inner_->blob(l, e) : FileExpertSource::blob(l, e);
    }
    bool fb_copy_blob(int64_t l, int64_t e, uint8_t* dst) {
        return inner_ != nullptr ? inner_->copy_blob(l, e, dst)
                                 : FileExpertSource::copy_blob(l, e, dst);
    }
    void fb_prefetch(int64_t l, const int64_t* e, int64_t n) {
        if (inner_ != nullptr) inner_->prefetch(l, e, n);
        else FileExpertSource::prefetch(l, e, n);
    }
    void fb_warm(int64_t l, const int64_t* e, int64_t n) {
        if (inner_ != nullptr) inner_->warm(l, e, n);
        else FileExpertSource::warm(l, e, n);
    }
    void fb_note(const uint8_t* b, void* s) {
        if (inner_ != nullptr) inner_->note_async_read(b, s);
        else FileExpertSource::note_async_read(b, s);
    }
    int64_t fb_reads() const {
        return (inner_ != nullptr ? inner_->reads() : FileExpertSource::reads());
    }
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
    std::vector<uint8_t> held_bits_;  ///< HELLO_REPLY field 6 (empty = ask)
    int64_t tier_skipped_ = 0;  ///< pairs answered from inner_ via the bitset
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

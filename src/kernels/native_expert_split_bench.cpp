// src/kernels/native_expert_split_bench.cpp - scratch (T3): split one layer's expert GEMMs by intermediate rows
// across two GPUs and time each side's partial product.
//
// usage: native_expert_split_bench <layer> <groups> <tokens> <iters> <mode> <out_prefix|-> <shard.gguf...>
//
// A sliced expert blob is a valid native expert blob of geometry (n_embd=2560, n_ff=r): gate/up output rows are
// contiguous; the down matrix's input columns are block-aligned (r a multiple of 32) and repacked once on the host.
// Every side calls the same native_expert_grouped the daemon calls, so the partial product's kernel is the deployed
// kernel, not a model of it.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

namespace {

constexpr int64_t H = 2560, FF = 640;
const int kRatios[] = {64, 128, 160, 192, 256};   // 10 / 20 / 25 / 30 / 40 % of the 640 intermediate rows

[[noreturn]] void die(const char* what) {
    std::fprintf(stderr, "native_expert_split_bench: %s\n", what);
    std::exit(2);
}

struct Dev {
    uint8_t* blob = nullptr;
    float* x = nullptr;             // NTOK * H
    void* xq = nullptr;             // NTOK q8_1 rows
    void* scratch = nullptr;
    float* out = nullptr;
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *ng = nullptr, *dst = nullptr, *tok = nullptr;
};

struct Res {
    double us = 0, d2h_us = 0, h2d_us = 0, h2d_res_us = 0, bytes = 0;
    std::vector<float> out;
};

int g_layer = 47;

void fill_entries(int G, int T, std::vector<unsigned long long>& ptr, std::vector<int32_t>& start,
                  std::vector<int32_t>& dst, std::vector<int32_t>& tok, int NTOK, const uint8_t* blob_dev,
                  size_t bytes) {
    const int NE = G * T;
    ptr.resize(G);
    start.resize(G + 1);
    dst.resize(NE);
    tok.resize(NE);
    for (int g = 0; g < G; ++g) {
        ptr[g] = (unsigned long long) (blob_dev + (size_t) g * bytes);
        start[g] = g * T;
        for (int j = 0; j < T; ++j) {
            dst[g * T + j] = g * T + j;
            tok[g * T + j] = (g * 3 + j) % NTOK;
        }
    }
    start[G] = NE;
}

// Builds the [r0,r1) slice of every expert: gate/up rows contiguous, down columns block-aligned and repacked.
std::vector<uint8_t> slice_blobs(const std::vector<uint8_t>& full, int G, const K::NativeExpertLayout& Lf,
                                 const K::NativeExpertLayout& Ls, int r0, int r1) {
    const int nff = r1 - r0;
    std::vector<uint8_t> out((size_t) G * Ls.bytes, 0);
    const size_t gu = Lf.gu_row;
    const size_t dslice = Ls.d_row;
    for (int g = 0; g < G; ++g) {
        const uint8_t* src = full.data() + (size_t) g * Lf.bytes;
        uint8_t* dst = out.data() + (size_t) g * Ls.bytes;
        std::memcpy(dst, src + (size_t) r0 * gu, (size_t) nff * gu);                                  // gate rows
        std::memcpy(dst + Ls.up_off, src + Lf.up_off + (size_t) r0 * gu, (size_t) nff * gu);           // up rows
        for (int64_t row = 0; row < H; ++row)                                                          // down columns
            std::memcpy(dst + Ls.down_off + (size_t) row * dslice,
                        src + Lf.down_off + (size_t) row * Lf.d_row + (size_t) (r0 / 32) * 18, dslice);
    }
    return out;
}

void release(Dev& d) {
    if (d.blob) cudaFree(d.blob);
    if (d.x) cudaFree(d.x);
    if (d.xq) cudaFree(d.xq);
    if (d.scratch) cudaFree(d.scratch);
    if (d.out) cudaFree(d.out);
    if (d.ptr) cudaFree(d.ptr);
    if (d.start) cudaFree(d.start);
    if (d.ng) cudaFree(d.ng);
    if (d.dst) cudaFree(d.dst);
    if (d.tok) cudaFree(d.tok);
    d = Dev{};
}

// Runs one side (r0..r1) of every expert and returns its timing and output.
Res run_side(const std::vector<uint8_t>& full, const K::NativeExpertLayout& Lf, int G, int T, int NTOK,
             const std::vector<float>& x, const std::vector<uint8_t>& xq_host, int r0, int r1, int iters,
             int mode, const std::string& blobfile, cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1) {
    Res res0;
    const int64_t gu = Lf.gu_type, dt = Lf.d_type;
    const K::NativeExpertLayout L = K::native_expert_layout((int) gu, (int) dt, H, r1 - r0);
    std::vector<uint8_t> blobs = (r0 == 0 && r1 == FF) ? full : slice_blobs(full, G, Lf, L, r0, r1);

    const int NE = G * T;
    std::vector<unsigned long long> ptr;
    std::vector<int32_t> start, dst, tok;

    Dev d;
    {
        size_t fz = 0, tt = 0;
        cudaMemGetInfo(&fz, &tt);
        std::printf("VRAM free=%.1f MiB total=%.1f MiB need=%.1f MiB\n", fz / 1048576.0, tt / 1048576.0,
                    blobs.size() / 1048576.0);
    }
    if (cudaMalloc((void**) &d.blob, blobs.size()) != cudaSuccess) {
        std::printf("SKIP r0=%d r1=%d: cudaMalloc blob failed (%s)\n", r0, r1, cudaGetErrorString(cudaGetLastError()));
        return res0;
    }
    if (cudaMalloc((void**) &d.x, x.size() * sizeof(float)) != cudaSuccess) die("cudaMalloc x");
    if (cudaMalloc(&d.xq, xq_host.size()) != cudaSuccess) die("cudaMalloc xq");
    if (cudaMalloc(&d.scratch, K::native_expert_scratch_bytes(NE, L.n_ff)) != cudaSuccess) die("cudaMalloc scratch");
    if (cudaMalloc((void**) &d.out, (size_t) NE * H * sizeof(float)) != cudaSuccess) die("cudaMalloc out");
    if (cudaMalloc((void**) &d.ptr, (size_t) G * sizeof(unsigned long long)) != cudaSuccess) die("cudaMalloc ptr");
    if (cudaMalloc(&d.start, (size_t) (G + 1) * 4) != cudaSuccess) die("cudaMalloc start");
    if (cudaMalloc(&d.ng, 4) != cudaSuccess) die("cudaMalloc ng");
    if (cudaMalloc(&d.dst, (size_t) NE * 4) != cudaSuccess) die("cudaMalloc dst");
    if (cudaMalloc(&d.tok, (size_t) NE * 4) != cudaSuccess) die("cudaMalloc tok");
    cudaMemcpy(d.blob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d.x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d.xq, xq_host.data(), xq_host.size(), cudaMemcpyHostToDevice);
    fill_entries(G, T, ptr, start, dst, tok, NTOK, d.blob, L.bytes);   // DEVICE addresses
    cudaMemcpy(d.ptr, ptr.data(), (size_t) G * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d.start, start.data(), (size_t) (G + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d.ng, &G, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d.dst, dst.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d.tok, tok.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);

    auto run = [&] {
        K::native_expert_set_mode(mode, 0);
        K::native_expert_grouped(L, d.ptr, d.start, d.ng, d.dst, d.tok, G, NE, d.xq, d.scratch, d.out, s);
    };
    for (int i = 0; i < 10; ++i) run();
    float best = 1e30f;
    for (int round = 0; round < 3; ++round) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i) run();
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, ms);
    }
    Res res;
    res.us = 1e3 * (double) best / iters;
    res.bytes = (double) G * (double) L.bytes;
    res.out.resize((size_t) NE * H);
    cudaMemcpyAsync(res.out.data(), d.out, res.out.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    // phase split: 1 = gate/up + SwiGLU + q8_1, 2 = down only
    double ph_us[3] = {0, 0, 0};
    for (int ph = 1; ph <= 2; ++ph) {
        K::native_expert_set_mode(mode, ph);
        for (int i = 0; i < 5; ++i)
            K::native_expert_grouped(L, d.ptr, d.start, d.ng, d.dst, d.tok, G, NE, d.xq, d.scratch, d.out, s);
        float bp = 1e30f;
        for (int round = 0; round < 3; ++round) {
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i)
                K::native_expert_grouped(L, d.ptr, d.start, d.ng, d.dst, d.tok, G, NE, d.xq, d.scratch, d.out, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            bp = std::min(bp, ms);
        }
        ph_us[ph] = 1e3 * (double) bp / iters;
    }
    K::native_expert_set_mode(mode, 0);

    // result D2H (what the daemon's reply copy costs)
    float bd = 1e30f;
    for (int round = 0; round < 3; ++round) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i)
            cudaMemcpyAsync(res.out.data(), d.out, res.out.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        bd = std::min(bd, ms);
    }
    res.d2h_us = 1e3 * (double) bd / iters;

    // activation H2D (the daemon's q8_1 upload)
    float bh = 1e30f;
    for (int round = 0; round < 3; ++round) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i) cudaMemcpyAsync(d.xq, xq_host.data(), xq_host.size(), cudaMemcpyHostToDevice, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        bh = std::min(bh, ms);
    }
    res.h2d_us = 1e3 * (double) bh / iters;

    // result H2D: the fast card pulling the P40's partial in for the combine
    float br = 1e30f;
    for (int round = 0; round < 3; ++round) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i)
            cudaMemcpyAsync(d.out, res.out.data(), res.out.size() * sizeof(float), cudaMemcpyHostToDevice, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        br = std::min(br, ms);
    }
    res.h2d_res_us = 1e3 * (double) br / iters;

    if (!blobfile.empty()) {
        FILE* f = std::fopen(blobfile.c_str(), "wb");
        if (!f) die("fopen output");
        std::fwrite(res.out.data(), sizeof(float), res.out.size(), f);
        std::fclose(f);
    }
    std::printf("SIDE r0=%d r1=%d nff=%d us=%.1f gbs=%.1f gu_us=%.1f down_us=%.1f d2h_us=%.1f h2d_us=%.1f "
                "h2d_res_us=%.1f n=%zu\n",
                r0, r1, (int) L.n_ff, res.us, res.bytes / res.us / 1e3, ph_us[1], ph_us[2], res.d2h_us, res.h2d_us,
                res.h2d_res_us, res.out.size());
    release(d);
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 9) {
        std::fprintf(stderr, "usage: native_expert_split_bench <layer> <groups> <tokens> <iters> <mode> "
                             "<out_prefix|-> <all|lo> <shard.gguf...>\n");
        return 2;
    }
    const int layer = std::atoi(argv[1]), G = std::atoi(argv[2]), T = std::atoi(argv[3]);
    const int iters = std::atoi(argv[4]), mode = std::atoi(argv[5]);
    const std::string prefix = argv[6];
    const bool all_sides = std::string(argv[7]) != "lo";
    if (G < 1 || T < 1 || iters < 1) die("bad shape");
    const int NTOK = std::max(8, T);

    std::unique_ptr<strata::GgufFile> gguf;
    const strata::TensorInfo* t[3] = {};
    const std::string base = "blk." + std::to_string(layer) + ".ffn_";
    for (int i = 8; i < argc && !gguf; ++i) {
        try {
            auto f = std::make_unique<strata::GgufFile>(argv[i]);
            const strata::TensorInfo* a = f->find(base + "gate_exps.weight");
            const strata::TensorInfo* b = f->find(base + "up_exps.weight");
            const strata::TensorInfo* c = f->find(base + "down_exps.weight");
            if (!a || !b || !c) continue;
            t[0] = a;
            t[1] = b;
            t[2] = c;
            gguf = std::move(f);
            std::printf("FOUND %s\n", argv[i]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "skip %s: %s\n", argv[i], e.what());
        }
    }
    if (!gguf) die("no shard with the layer's expert tensors");

    cpu::NativeFmt ff;
    std::string err;
    if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, ff, err)) die(err.c_str());
    const K::NativeExpertLayout Lf = K::native_expert_layout(ff.gu_type, ff.d_type, H, FF);
    std::printf("LAYER %d gu=%d down=%d blob=%zu gu_row=%zu d_row=%zu\n", layer, ff.gu_type, ff.d_type, Lf.bytes,
                Lf.gu_row, Lf.d_row);

    // full blobs, exactly as native_expert_bench builds them
    std::vector<uint8_t> full((size_t) G * Lf.bytes);
    const size_t dsz = ff.bytes - ff.down_off;
    const size_t drow = dsz / H;
    for (int g = 0; g < G; ++g) {
        const size_t E = (size_t) ((g * 37 + 5) % 256);
        uint8_t* b = full.data() + (size_t) g * Lf.bytes;
        std::memcpy(b, gguf->tensor_data(*t[0]) + E * ff.up_off, ff.up_off);
        std::memcpy(b + Lf.up_off, gguf->tensor_data(*t[1]) + E * ff.up_off, ff.up_off);
        for (int64_t r = 0; r < H; ++r)
            std::memcpy(b + Lf.down_off + (size_t) r * Lf.d_row, gguf->tensor_data(*t[2]) + E * dsz + (size_t) r * drow,
                        Lf.d_row);
    }

    std::mt19937 rng(17 + layer);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NTOK * H);
    for (auto& v : x) v = nd(rng);
    std::vector<uint8_t> xq((size_t) NTOK * (H / 32) * 36);

    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    {
        float* dx = nullptr;
        void* dxq = nullptr;
        if (cudaMalloc((void**) &dx, x.size() * sizeof(float)) != cudaSuccess) die("cudaMalloc dx");
        if (cudaMalloc(&dxq, xq.size()) != cudaSuccess) die("cudaMalloc dxq");
        cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
        K::quantize_q8_1_rows(dx, NTOK, H, dxq, s);
        cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(s);
        cudaFree(dx);
        cudaFree(dxq);
    }

    std::printf("SHAPE G=%d T=%d NE=%d NTOK=%d iters=%d mode=%d\n", G, T, G * T, NTOK, iters, mode);
    auto save = [&](const char* what, int r) {
        return prefix == "-" ? std::string() : prefix + "." + what + ".r" + std::to_string(r) + ".bin";
    };

    Res whole;
    if (all_sides) whole = run_side(full, Lf, G, T, NTOK, x, xq, 0, (int) FF, iters, mode, save("whole", 0), s, e0, e1);
    for (int r : kRatios) {
        const Res lo = run_side(full, Lf, G, T, NTOK, x, xq, 0, r, iters, mode, save("lo", r), s, e0, e1);
        if (!all_sides) continue;
        const Res hi = run_side(full, Lf, G, T, NTOK, x, xq, r, (int) FF, iters, mode, save("hi", r), s, e0, e1);
        size_t bitwise = 0;
        double worst = 0, sum = 0;
        size_t n = whole.out.size();
        if (lo.out.empty() || hi.out.empty() || whole.out.empty()) {
            std::printf("CHECK r=%d skipped (a side was skipped)\n", r);
            continue;
        }
        for (size_t i = 0; i < n; ++i) {
            const float a = lo.out[i] + hi.out[i], b = whole.out[i];
            if (std::memcmp(&a, &b, 4) == 0) ++bitwise;
            const double d = (double) a - (double) b;
            if (std::fabs(d) > worst) worst = std::fabs(d);
            sum += d * d;
        }
        std::printf("CHECK r=%d same_gpu rms=%.3e max_abs=%.3e bitwise=%zu/%zu\n", r, std::sqrt(sum / (double) n),
                    worst, bitwise, n);
    }
    std::printf("DONE\n");
    return 0;
}

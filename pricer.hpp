// Tariff-refund claim pricer: Monte Carlo over stochastic time-to-resolution.
// PV per path = 1{win} * face * recovery * exp(-rate * T) - carry * (1 - exp(-rate * T)) / rate,
// T ~ Exponential | Weibull. Carry = legal/admin cost per year while unresolved, paid win or lose.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tariff {

enum class Dist { Exponential, Weibull };

struct Params {
    double face     = 1'000'000; // claim face value
    double p_win    = 0.60;      // P(refund granted)
    double recovery = 1.00;      // fraction of face paid on win
    double rate     = 0.05;      // continuous discount rate (per year)
    Dist   dist     = Dist::Exponential;
    double scale    = 2.0;       // Exponential: mean years. Weibull: scale (lambda)
    double shape    = 1.0;       // Weibull: k (ignored for Exponential)
    double carry    = 0.0;       // legal/admin cost per year while unresolved
    std::size_t paths = 100'000;
    std::uint64_t seed = 42;
};

struct Result {
    double mean = 0, stddev = 0, p5 = 0, p50 = 0, p95 = 0;
    double mean_discount = 0;    // E[exp(-rate*T)], used for implied-probability inference
    double mean_carry = 0;       // E[PV of carry cost], subtracted from every path
};

// xoshiro256** : ~1ns/draw, no heap, no branches. Seeded via splitmix64.
struct Rng {
    std::uint64_t s[4];
    explicit Rng(std::uint64_t seed) {
        for (auto& x : s) {
            std::uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            x = z ^ (z >> 31);
        }
    }
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t next() {
        const std::uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    double uniform() { return ((next() >> 11) + 1) * 0x1.0p-53; } // (0,1]: safe for log()
};

// Inverse-CDF sample of time-to-resolution.
inline double draw_time(const Params& p, double u) {
    const double e = -std::log(u);                       // Exp(1)
    return p.dist == Dist::Exponential ? p.scale * e : p.scale * std::pow(e, 1.0 / p.shape);
}

// Fills `pv` (resized to p.paths) with discounted cash flows and returns summary stats.
// Optional `tt` receives each path's resolution time. Caller owns the buffers so repeated
// pricing does no per-path allocation.
inline Result simulate(const Params& p, std::vector<double>& pv, std::vector<double>* tt = nullptr) {
    pv.resize(p.paths);
    if (tt) tt->resize(p.paths);
    const double payoff = p.face * p.recovery;
    int nth = 1;
    #ifdef _OPENMP
    nth = omp_get_max_threads();
    #endif
    // Per-thread partials summed in thread order: bit-identical results for a fixed thread count
    // (an OpenMP reduction would combine in arbitrary order).
    std::vector<double> part(4 * nth, 0.0);

    // Each OpenMP thread gets its own RNG stream and a contiguous slice of paths.
    #pragma omp parallel num_threads(nth)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        Rng rng(p.seed + 0x1000ull * tid);
        double sum = 0, sumsq = 0, sumdisc = 0, sumcarry = 0;
        const std::size_t lo = p.paths * tid / nth, hi = p.paths * (tid + 1) / nth;
        for (std::size_t i = lo; i < hi; ++i) {
            const double T = draw_time(p, rng.uniform());
            const double disc = std::exp(-p.rate * T);
            const double carry = p.carry * (p.rate > 0 ? (1.0 - disc) / p.rate : T); // PV of continuous cost over [0,T]
            const double v = ((rng.uniform() <= p.p_win) ? payoff * disc : 0.0) - carry;
            pv[i] = v; sum += v; sumsq += v * v; sumdisc += disc; sumcarry += carry;
            if (tt) (*tt)[i] = T;
        }
        part[4 * tid] = sum; part[4 * tid + 1] = sumsq; part[4 * tid + 2] = sumdisc; part[4 * tid + 3] = sumcarry;
    }
    double sum = 0, sumsq = 0, sumdisc = 0, sumcarry = 0;
    for (int t = 0; t < nth; ++t) {
        sum += part[4 * t]; sumsq += part[4 * t + 1]; sumdisc += part[4 * t + 2]; sumcarry += part[4 * t + 3];
    }

    Result r;
    const double n = static_cast<double>(p.paths);
    r.mean = sum / n;
    r.stddev = std::sqrt(std::max(0.0, sumsq / n - r.mean * r.mean));
    r.mean_discount = sumdisc / n;
    r.mean_carry = sumcarry / n;
    // Percentiles on a scratch copy so `pv` stays in path order for the caller (convergence plots, CSV).
    std::vector<double> scratch(pv);
    auto pct = [&](double q) {
        auto k = scratch.begin() + static_cast<std::ptrdiff_t>(q * (n - 1));
        std::nth_element(scratch.begin(), k, scratch.end());
        return *k;
    };
    r.p5 = pct(0.05); r.p50 = pct(0.50); r.p95 = pct(0.95);
    return r;
}

// Market quotes the claim at `price_frac` of face. Under E[PV] = price, solve for P(win):
//   price = p_win * face * recovery * E[disc] - E[carry]
inline double implied_prob(double price_frac, const Params& p, const Result& r) {
    return (price_frac * p.face + r.mean_carry) / (p.face * p.recovery * r.mean_discount);
}

// Mean-variance certainty equivalent: hold iff CE > offer. `risk` = penalty per unit stddev.
// ponytail: linear risk penalty; swap for exponential-utility CE if calibrating to real risk limits.
inline double certainty_equiv(const Result& r, double risk) { return r.mean - risk * r.stddev; }

} // namespace tariff

// Self-check: analytic limits the simulator must hit. Run: make test
#include "pricer.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
using namespace tariff;

int main() {
    std::vector<double> pv;
    Params p; p.paths = 400'000;

    // 1. rate=0 => E[PV] = p_win*face, within 4 standard errors
    p.rate = 0; Result r = simulate(p, pv);
    double se = r.stddev / std::sqrt((double)p.paths);
    assert(std::fabs(r.mean - p.p_win * p.face) < 4 * se);

    // 2. Exponential: E[exp(-rT)] = 1/(1 + r*scale)
    p.rate = 0.05; r = simulate(p, pv);
    assert(std::fabs(r.mean_discount - 1 / (1 + p.rate * p.scale)) < 2e-3);

    // 3. implied probability round-trips (up to MC noise: win-path discount vs all-path discount)
    assert(std::fabs(implied_prob(r.mean / p.face, p, r) - p.p_win) < 2e-3);

    // 4. Weibull k=1 == Exponential
    Params w = p; w.dist = Dist::Weibull; w.shape = 1.0;
    Result rw = simulate(w, pv);
    assert(std::fabs(rw.mean_discount - r.mean_discount) < 1e-12);

    // 5. deterministic for fixed seed, and CE monotone in risk
    Result r2 = simulate(p, pv);
    assert(r2.mean == r.mean && r2.p95 == r.p95);
    assert(certainty_equiv(r, 0) > certainty_equiv(r, 1));

    std::puts("all checks passed");
}

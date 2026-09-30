#include "pricer.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

using namespace tariff;

static void usage() {
    std::puts("tariff-pricer [options]\n"
              "  --face F        claim face value         (1000000)\n"
              "  --p P           P(refund granted)        (0.60)\n"
              "  --recovery R    fraction paid on win     (1.0)\n"
              "  --rate r        discount rate            (0.05)\n"
              "  --dist exp|weibull                       (exp)\n"
              "  --scale S       mean yrs / Weibull scale (2.0)\n"
              "  --shape K       Weibull shape            (1.0)\n"
              "  --paths N       Monte Carlo paths        (100000)\n"
              "  --seed S        RNG seed                 (42)\n"
              "  --market M      observed price/face -> implied P(win)\n"
              "  --offer O       cash offer/face for sell-vs-hold\n"
              "  --risk L        stddev penalty for CE    (0.5)\n"
              "  --csv FILE      dump PV samples for plot.py");
}

int main(int argc, char** argv) {
    Params p; double market = -1, offer = -1, risk = 0.5; const char* csv = nullptr;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
        if      (is("--face"))     p.face = std::atof(argv[++i]);
        else if (is("--p"))        p.p_win = std::atof(argv[++i]);
        else if (is("--recovery")) p.recovery = std::atof(argv[++i]);
        else if (is("--rate"))     p.rate = std::atof(argv[++i]);
        else if (is("--dist"))     p.dist = std::strcmp(argv[++i], "weibull") == 0 ? Dist::Weibull : Dist::Exponential;
        else if (is("--scale"))    p.scale = std::atof(argv[++i]);
        else if (is("--shape"))    p.shape = std::atof(argv[++i]);
        else if (is("--paths"))    p.paths = std::strtoull(argv[++i], nullptr, 10);
        else if (is("--seed"))     p.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (is("--market"))   market = std::atof(argv[++i]);
        else if (is("--offer"))    offer = std::atof(argv[++i]);
        else if (is("--risk"))     risk = std::atof(argv[++i]);
        else if (is("--csv"))      csv = argv[++i];
        else { usage(); return 1; }
    }
    if (p.paths == 0 || p.p_win < 0 || p.p_win > 1 || p.scale <= 0 || p.shape <= 0) { usage(); return 1; }

    std::vector<double> pv;
    auto t0 = std::chrono::steady_clock::now();
    Result r = simulate(p, pv);
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();

    std::printf("paths=%zu  dist=%s  scale=%.2f shape=%.2f  rate=%.3f  p_win=%.3f\n",
                p.paths, p.dist == Dist::Weibull ? "weibull" : "exp", p.scale, p.shape, p.rate, p.p_win);
    std::printf("sim time   %.1f us  (%.1f M paths/s)\n", us, p.paths / us);
    std::printf("E[PV]      %.2f   (%.4f of face)\n", r.mean, r.mean / p.face);
    std::printf("stddev     %.2f\n", r.stddev);
    std::printf("p5/p50/p95 %.2f / %.2f / %.2f\n", r.p5, r.p50, r.p95);
    std::printf("E[disc]    %.4f\n", r.mean_discount);

    if (market >= 0)
        std::printf("implied P(win) at market %.3f: %.4f\n", market, implied_prob(market, p, r));

    std::printf("\nsell-vs-hold (CE = mean - risk*stddev), breakeven offer as fraction of face:\n");
    for (double l : {0.0, 0.25, 0.5, 1.0, 2.0})
        std::printf("  risk=%.2f  breakeven=%.4f\n", l, certainty_equiv(r, l) / p.face);
    if (offer >= 0) {
        double ce = certainty_equiv(r, risk) / p.face;
        std::printf("offer %.4f vs CE %.4f (risk=%.2f) -> %s\n", offer, ce, risk, ce > offer ? "HOLD" : "SELL");
    }

    if (csv) {
        std::ofstream f(csv);
        f << "pv\n";
        for (double v : pv) f << v << '\n';
        std::printf("wrote %s\n", csv);
    }
}

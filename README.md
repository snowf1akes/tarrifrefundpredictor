# Tariff-Refund Claim Pricer

Monte Carlo pricer for illiquid contingent claims: a tariff refund that pays `face * recovery`
at a random future resolution time `T`, with probability `p_win`, else nothing.

```
PV = 1{win} * face * recovery * exp(-rate * T),   T ~ Exponential(scale) | Weibull(scale, shape)
```

C++17, header-only engine, OpenMP hot loop, xoshiro256** RNG with inverse-CDF sampling.
No allocation per path; ~50M paths/s on a laptop.

## What it does

- **Discounted expected-value pricing**: mean, stddev, p5/p50/p95 of PV over 100K+ paths.
- **Stochastic time-to-resolution**: Exponential or Weibull, sampled by inverse CDF.
- **Implied probability inference**: given an observed market price as a fraction of face,
  solve `price = p_win * recovery * E[exp(-rT)]` for `p_win`.
- **Sell-vs-hold decision boundary**: certainty equivalent `CE = mean - risk * stddev`;
  hold iff `CE > offer`. Prints the breakeven offer across risk penalties.

## Build

```bash
g++ -std=c++17 -O3 -march=native -fopenmp main.cpp -o tariff-pricer
```

Or `make` / `make test` on Linux, macOS, MSYS2. On Windows without a compiler: `scoop install gcc`.

## Run

```bash
./tariff-pricer --paths 1000000 --p 0.6 --rate 0.05 --dist weibull --scale 2.5 --shape 1.5 \
                --market 0.45 --offer 0.40 --risk 0.5 --csv samples.csv
python plot.py samples.csv        # matplotlib histogram -> pv_hist.png
```

Example output:

```
paths=1000000  dist=exp  scale=2.00 shape=1.00  rate=0.050  p_win=0.600
sim time   20813.0 us  (48.0 M paths/s)
E[PV]      545278.89   (0.5453 of face)
stddev     449874.13
p5/p50/p95 0.00 / 835303.88 / 991272.38
E[disc]    0.9090
implied P(win) at market 0.450: 0.4950

sell-vs-hold (CE = mean - risk*stddev), breakeven offer as fraction of face:
  risk=0.00  breakeven=0.5453
  risk=0.25  breakeven=0.4328
  risk=0.50  breakeven=0.3203
  risk=1.00  breakeven=0.0954
  risk=2.00  breakeven=-0.3545
offer 0.4000 vs CE 0.3203 (risk=0.50) -> SELL
```

All flags: `./tariff-pricer --help`.

## Test

```bash
g++ -std=c++17 -O3 -fopenmp test.cpp -o run-test && ./run-test
```

Checks the simulator against analytic limits: `E[PV] = p_win * face` at zero rate,
`E[exp(-rT)] = 1/(1 + r*scale)` for Exponential, Weibull `k=1` equals Exponential,
implied-probability round-trip, and seed determinism.

## Layout

| File | Purpose |
|---|---|
| `pricer.hpp` | engine: `Params`, `simulate`, `implied_prob`, `certainty_equiv` |
| `main.cpp` | CLI |
| `test.cpp` | assert-based self-check |
| `plot.py` | histogram of PV samples |

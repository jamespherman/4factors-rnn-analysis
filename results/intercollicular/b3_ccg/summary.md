# B3 within-SC CCGs: 13578 directed pairs (different channels, coincidence <= 0.1)

Peak/trough: Poisson p < 0.001 in 1-4 ms vs jitter, 1-4 ms vs -4..-1 ms asymmetric (binomial p < 0.001) and sharper than the 6-10 ms shoulders. Broad = jitter-significant but not sharp (fast common input). Powered: min detectable STG <= 0.01.

| pre -> post | pairs | powered | peaks (all / powered) | troughs (all / powered) | median STG of peaks |
|---|---|---|---|---|---|
| E -> E | 1740 | 1465 | 14 / 9 (0.6%) | 0 / 0 | 0.022 |
| E -> I | 2921 | 2449 | 24 / 21 (0.9%) | 0 / 0 | 0.013 |
| I -> E | 2921 | 2791 | 21 / 21 (0.8%) | 1 / 0 | 0.006 |
| I -> I | 5996 | 5599 | 64 / 62 (1.1%) | 7 / 5 | 0.008 |

## Distance dependence (powered pairs, all classes)
- 0-100 um: 758 pairs, sharp peaks 1.2%, broad 11.7%, troughs 0.3%
- 100-200 um: 1468 pairs, sharp peaks 1.0%, broad 10.7%, troughs 0.0%
- 200-400 um: 2541 pairs, sharp peaks 0.6%, broad 10.0%, troughs 0.0%
- 400-2000 um: 7537 pairs, sharp peaks 1.0%, broad 7.1%, troughs 0.0%

## Planned contrast: classic -> non-classic peaks, by whether the non-classic target responds at ipsi locations
- ipsi-responsive targets: 21/2116 (1.0%); others: 0/333 (0.0%); Fisher p=0.1
- median spikes (pre, post): ipsi-responsive targets 24160, 28184; others 21104, 32358; median min-detectable STG 0.0046 vs 0.0048
- units receiving >= 1 classic peak: ipsi-responsive 14/161, others 0/27

## Troughs (listed; not interpreted)
- Feynman_08_15 100(I) -> 2(I): STG -0.0076, p 6e-07, dist 1250 um
- Feynman_08_15 109(I) -> 107(I): STG -0.0066, p 9e-07, dist 50 um
- Feynman_08_15 111(I) -> 8(I): STG -0.0055, p 0.0002, dist 950 um
- Feynman_08_15 111(I) -> 17(I): STG -0.0155, p 6e-05, dist 900 um
- Feynman_08_15 111(I) -> 21(I): STG -0.0163, p 1e-06, dist 800 um
- Feynman_08_15 111(I) -> 42(I): STG -0.0140, p 5e-11, dist 550 um
- 08_14 32(I) -> 34(I): STG -0.0150, p 3e-11, dist 50 um
- 08_14 32(I) -> 71(E): STG -0.0100, p 0.0008, dist 600 um

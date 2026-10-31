# Stage 0a: hand-built architectures at matched behaviour (08-20 sequence, recorded unit/trial counts)

Monkey targets: {'p_right': 0.3603238866396761, 'p_high': 0.5587044534412956, 'rt': 208.766666666634}

## Calibration (engaged fraction w, side bias)
- A: w 0.300, bias 0.000, p_right 0.376, p_high 0.557, rt 200.665, completed 0.999
- B: w 0.300, bias 0.000, p_right 0.416, p_high 0.568, rt 208.200, completed 0.999
- C: w 0.100, bias 0.000, p_right 0.412, p_high 0.558, rt 214.288, completed 0.998, c_gain 1.100, theta 0.750

## Signature distributions (mean +- SD over draws) and separation
- S1_delay: A: +1.29 +- 0.97 | B: +0.12 +- 1.25 | C: +0.13 +- 0.98 || BvC E|logLR| 0.0, AvB E|logLR| 0.5, AvC E|logLR| 0.7
- S1_presac: A: +1.01 +- 0.86 | B: +0.54 +- 1.16 | C: +0.09 +- 1.11 || BvC E|logLR| 0.1, AvB E|logLR| 0.3, AvC E|logLR| 0.4
- S2_mean: A: +0.01 +- 0.01 | B: -0.00 +- 0.01 | C: -0.00 +- 0.01 || BvC E|logLR| 0.1, AvB E|logLR| 0.6, AvC E|logLR| 0.8
- S2_t: A: +1.50 +- 1.25 | B: -0.03 +- 1.15 | C: -0.26 +- 1.04 || BvC E|logLR| 0.1, AvB E|logLR| 1.0, AvC E|logLR| 1.8
- S7_delay_t: A: +1.83 +- 0.89 | B: +0.14 +- 1.07 | C: -0.24 +- 1.00 || BvC E|logLR| 0.1, AvB E|logLR| 1.1, AvC E|logLR| 2.5
- S8_delay_absz: A: +13.62 +- 1.25 | B: +0.81 +- 0.08 | C: +0.79 +- 0.09 || BvC E|logLR| 0.0, AvB E|logLR| 26.0, AvC E|logLR| 26.0
- S7_presac_t: A: +0.44 +- 0.87 | B: -0.69 +- 1.07 | C: -0.03 +- 1.07 || BvC E|logLR| 0.2, AvB E|logLR| 0.7, AvC E|logLR| 0.3
- S8_presac_absz: A: +7.58 +- 0.69 | B: +0.78 +- 0.09 | C: +0.83 +- 0.09 || BvC E|logLR| 0.2, AvB E|logLR| 26.6, AvC E|logLR| 26.6
- S9_rt_value_rho: A: -0.34 +- 0.05 | B: -0.35 +- 0.06 | C: +0.01 +- 0.05 || BvC E|logLR| 28.4, AvB E|logLR| 0.0, AvC E|logLR| 28.8

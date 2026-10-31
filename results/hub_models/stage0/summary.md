# Stage 0a: hand-built architectures at matched behaviour (08-20 sequence, recorded unit/trial counts)

Monkey targets: {'p_right': 0.3603238866396761, 'p_high': 0.5587044534412956, 'rt': 208.766666666634}

## Calibration (engaged fraction w, side bias)
- A: w 0.300, bias 0.000, p_right 0.376, p_high 0.551, rt 197.317, completed 0.999
- B: w 0.300, bias 0.000, p_right 0.392, p_high 0.557, rt 205.725, completed 0.999
- C: w 0.100, bias 0.000, p_right 0.394, p_high 0.549, rt 210.265, completed 0.998, c_gain 1.100, theta 0.750

## Signature distributions (mean +- SD over draws) and separation
- S1_delay: A: +1.10 +- 0.80 | B: +0.32 +- 0.80 | C: +0.49 +- 0.80 || BvC E|logLR| 0.0, AvB E|logLR| 0.5, AvC E|logLR| 0.4
- S1_presac: A: +1.09 +- 1.03 | B: +0.39 +- 0.95 | C: +0.62 +- 1.17 || BvC E|logLR| 0.3, AvB E|logLR| 0.3, AvC E|logLR| 0.2
- S2_mean: A: +0.01 +- 0.01 | B: +0.00 +- 0.01 | C: -0.00 +- 0.01 || BvC E|logLR| 0.1, AvB E|logLR| 0.3, AvC E|logLR| 0.4
- S2_t: A: +1.17 +- 1.14 | B: +0.30 +- 0.98 | C: -0.04 +- 0.84 || BvC E|logLR| 0.1, AvB E|logLR| 0.3, AvC E|logLR| 1.3
- S7_delay_t: A: +1.80 +- 0.89 | B: -0.05 +- 1.18 | C: -0.13 +- 1.53 || BvC E|logLR| 0.3, AvB E|logLR| 1.1, AvC E|logLR| 0.9
- S8_delay_absz: A: +8.54 +- 0.73 | B: +0.83 +- 0.08 | C: +0.86 +- 0.10 || BvC E|logLR| 0.2, AvB E|logLR| 26.6, AvC E|logLR| 26.6
- S8_delay_excess: A: +7.75 +- 0.73 | B: +0.01 +- 0.08 | C: +0.06 +- 0.09 || BvC E|logLR| 0.2, AvB E|logLR| 26.6, AvC E|logLR| 26.6
- S8int_delay_excess: A: +0.11 +- 0.11 | B: +0.01 +- 0.10 | C: -0.01 +- 0.09 || BvC E|logLR| 0.1, AvB E|logLR| 0.6, AvC E|logLR| 1.0
- S7_presac_t: A: +0.58 +- 0.86 | B: -0.52 +- 1.04 | C: +0.36 +- 1.12 || BvC E|logLR| 0.5, AvB E|logLR| 0.6, AvC E|logLR| 0.2
- S8_presac_absz: A: +6.27 +- 0.55 | B: +0.87 +- 0.11 | C: +0.82 +- 0.08 || BvC E|logLR| 0.2, AvB E|logLR| 26.9, AvC E|logLR| 26.9
- S8_presac_excess: A: +5.47 +- 0.55 | B: +0.08 +- 0.11 | C: +0.02 +- 0.08 || BvC E|logLR| 0.2, AvB E|logLR| 26.9, AvC E|logLR| 26.9
- S8int_presac_excess: A: +0.04 +- 0.09 | B: +0.01 +- 0.08 | C: -0.01 +- 0.10 || BvC E|logLR| 0.2, AvB E|logLR| 0.1, AvC E|logLR| 0.1
- S9_rt_value_rho: A: -0.32 +- 0.04 | B: -0.32 +- 0.05 | C: +0.02 +- 0.05 || BvC E|logLR| 27.9, AvB E|logLR| 0.0, AvC E|logLR| 27.7

# Urgent-task simulations (srsClassic structure: reward tied to identity), 69 networks, 2026-11-01

Summary table: `summary.md`; per network: `per_network.csv`; figure: `tachometric.png`; raw analyses:
`../urgent_raw/*.json`. Plan: `plans/2026-10-31_urgent_task_plan.md`.

- Divergence locking (target- vs go-locked): no architecture difference (slopes -0.17 to -0.25; A vs B p = 0.21).
  The one-network hint from the test run did not hold.
- Capture-error trials: the rich-site rate is not suppressed in any architecture (ratio A 1.32, B 1.69, C 0.99);
  A < B (p = 0.003) but this is a difference of degree, not the suppressed-vs-intact contrast hypothesised.
- Value-gap dependence of the asymptote: A 0.25 > B 0.16 > C 0.01 (all p < 1e-5). Graded vs categorical again.
- S9 (graded delivery): A -0.15, B -0.16, C -0.03 (C differs, p ~1e-5), as in the delayed task.
- S8 identity code: A 0.74 vs hubs 0.18 (p ~3e-7), as in the delayed task.
- Outcome hubs (C) mostly failed to use value under time pressure (upper asymptote median 0.21; logistic fits
  unstable): the softmax WTA hub does not learn the urgent task well with this training budget.
Conclusion: in this task structure, the urgent design does not add a resolver-vs-hub signature beyond the identity
code; it again separates graded from categorical delivery. Not used in the R01 (srsClassic structure).

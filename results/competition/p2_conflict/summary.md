# P2 probability conflict (diagonal-target sessions; units with RF at a probability location)

## E units (n=50)
- (a) pre-target rate, prior at RF minus prior away: mean +0.18 sp/s (median +0.17), Wilcoxon p=0.089; low-reward cue +0.11, high-reward cue +0.23 (difference p=0.31)
- (b) target at a neighbour (elsewhere): prior-at-RF minus prior-away, by 50-ms window (sp/s; * p<0.01):
    -300: +0.15  | -250: -0.17  | -200: +0.11  | -150: +0.44  | -100: +0.23  | -50: +0.07  | -0: +0.16  | +50: +0.30  | +100: +0.12  | +150: +0.09  | +200: -0.10  | +250: -0.23  | +300: +0.05  | +350: -0.19 
    activity level on neighbour trials re pre-target baseline: -300: +0.2 | -250: +0.1 | -200: -0.0 | -150: +0.2 | -100: +0.1 | -50: +0.2 | -0: +0.2 | +50: +1.1 | +100: +2.0 | +150: +0.8 | +200: -0.0 | +250: -0.3 | +300: -0.2 | +350: -0.2
- (b2) target in the RF: expected minus unexpected:
    -300: -0.14  | -250: +0.05  | -200: +0.27  | -150: +0.80  | -100: +0.37  | -50: +0.14  | -0: +0.41  | +50: +1.14  | +100: -1.23  | +150: -0.68  | +200: -0.90  | +250: -0.34  | +300: -0.64  | +350: -0.35 

## I units (n=90)
- (a) pre-target rate, prior at RF minus prior away: mean +0.28 sp/s (median +0.06), Wilcoxon p=0.19; low-reward cue +0.21, high-reward cue +0.34 (difference p=0.38)
- (b) target at a neighbour (elsewhere): prior-at-RF minus prior-away, by 50-ms window (sp/s; * p<0.01):
    -300: +0.23  | -250: +0.38  | -200: +0.60  | -150: +0.13  | -100: +0.61  | -50: +0.36  | -0: +1.17* | +50: +0.23  | +100: +0.04  | +150: -0.12  | +200: -0.21  | +250: -0.03  | +300: -0.00  | +350: -0.11 
    activity level on neighbour trials re pre-target baseline: -300: -0.3 | -250: -0.2 | -200: -0.4 | -150: +0.0 | -100: +0.1 | -50: +0.5 | -0: +0.7 | +50: +2.2 | +100: +0.8 | +150: -1.2 | +200: -2.3 | +250: -1.9 | +300: -1.7 | +350: -1.1
- (b2) target in the RF: expected minus unexpected:
    -300: -0.28  | -250: -0.35  | -200: +0.22  | -150: +0.54  | -100: +0.35  | -50: +0.35  | -0: +0.29  | +50: +1.54* | +100: -3.01* | +150: -2.83* | +200: -1.71* | +250: -0.51  | +300: -0.64  | +350: -0.59 

## all units (n=140)
- (a) pre-target rate, prior at RF minus prior away: mean +0.24 sp/s (median +0.13), Wilcoxon p=0.047; low-reward cue +0.18, high-reward cue +0.30 (difference p=0.2)
- (b) target at a neighbour (elsewhere): prior-at-RF minus prior-away, by 50-ms window (sp/s; * p<0.01):
    -300: +0.20  | -250: +0.19  | -200: +0.43  | -150: +0.24  | -100: +0.47  | -50: +0.26  | -0: +0.81* | +50: +0.26  | +100: +0.07  | +150: -0.04  | +200: -0.17  | +250: -0.10  | +300: +0.02  | +350: -0.14 
    activity level on neighbour trials re pre-target baseline: -300: -0.1 | -250: -0.1 | -200: -0.3 | -150: +0.1 | -100: +0.1 | -50: +0.4 | -0: +0.5 | +50: +1.8 | +100: +1.3 | +150: -0.5 | +200: -1.5 | +250: -1.4 | +300: -1.2 | +350: -0.8
- (b2) target in the RF: expected minus unexpected:
    -300: -0.23  | -250: -0.21  | -200: +0.24  | -150: +0.63  | -100: +0.36  | -50: +0.28  | -0: +0.33  | +50: +1.39* | +100: -2.37* | +150: -2.07* | +200: -1.42* | +250: -0.45  | +300: -0.64  | +350: -0.51 

## (c) Single-trial RT ~ pre-target SC state (OLS, session fixed effects; log RT x100 ~ %)
- target at a low-probability location: n=880; z_prior (units with RF at the high-p location) coef -5.53% per SD, p=0.016
  z_target (units with RF at the target location) coef -0.82% per SD, p=0.83, n=259
- target at the high-probability location: n=1461; z_prior (units with RF at the high-p location) coef -5.35% per SD, p=1e-05

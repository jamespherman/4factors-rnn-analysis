# P1 behaviour: 5242 successful trials, 7 sessions

## Cell means (median RT ms / mean peak velocity deg/s)

reward:
                   rt     pv     n
monkey  reward                    
Feynman 0       256.6  543.1  1496
        1       248.5  551.9  1499
Newton  0       241.6  595.2  1104
        1       242.0  586.4  1143

prob:
                      rt     pv     n
monkey  prob                         
Feynman high       250.0  547.5  1874
        neighbour  250.0  550.2   747
        opposite   250.0  542.3   374
Newton  high       241.7  552.5  1403
        neighbour  240.7  699.6   561
        opposite   249.9  564.3   283

stim:
                    rt     pv    n
monkey  stim                      
Feynman bull_hi  250.9  534.7  999
        bull_lo  251.0  530.9  998
        face     233.4  581.8  499
        nonface  241.7  571.9  499
Newton  bull_hi  242.1  568.9  750
        bull_lo  241.7  563.8  749
        face     241.6  637.1  375
        nonface  241.7  642.1  373

## log_rt ~ factors (Feynman, n=2995); coefficients with p < 0.1 (log RT: x100 ~ % change)
                                                                                    coef      se       p
C(stim, Treatment("bull_lo"))[T.face]                                           -14.3149  2.5057  0.0000
C(stim, Treatment("bull_lo"))[T.nonface]                                         -8.9464  2.1007  0.0000
C(prob, Treatment("high"))[T.opposite]:C(stim, Treatment("bull_lo"))[T.bull_hi]  -4.6280  2.2482  0.0395
reward                                                                           -8.8035  1.5532  0.0000
reward:C(prob, Treatment("high"))[T.neighbour]                                    5.4454  2.2767  0.0168
reward:C(prob, Treatment("high"))[T.opposite]                                    17.8960  2.4758  0.0000
reward:C(stim, Treatment("bull_lo"))[T.face]                                      4.1941  2.3983  0.0803
reward:C(stim, Treatment("bull_lo"))[T.nonface]                                   5.5400  2.3320  0.0175

## log_rt ~ factors (Newton, n=2247); coefficients with p < 0.1 (log RT: x100 ~ % change)
                                                                                    coef      se       p
C(prob, Treatment("high"))[T.neighbour]                                          -3.5266  1.9576  0.0716
C(stim, Treatment("bull_lo"))[T.nonface]                                          2.9203  1.0545  0.0056
C(prob, Treatment("high"))[T.neighbour]:C(stim, Treatment("bull_lo"))[T.nonface] -3.7411  2.2422  0.0952
C(prob, Treatment("high"))[T.opposite]:C(stim, Treatment("bull_lo"))[T.nonface]  -5.0335  2.4979  0.0439
reward                                                                            1.9490  1.1143  0.0803
reward:C(prob, Treatment("high"))[T.neighbour]                                    3.4224  1.6383  0.0367
reward:C(prob, Treatment("high"))[T.opposite]                                     3.5000  2.1181  0.0985
reward:C(stim, Treatment("bull_lo"))[T.nonface]                                  -4.9739  1.2967  0.0001

## log_rt ~ factors (both monkeys, n=5242); coefficients with p < 0.1 (log RT: x100 ~ % change)
                                                                                    coef      se       p
C(prob, Treatment("high"))[T.neighbour]                                          -3.0247  1.4197  0.0331
C(stim, Treatment("bull_lo"))[T.face]                                            -8.9436  1.9256  0.0000
C(stim, Treatment("bull_lo"))[T.nonface]                                         -3.9372  1.5089  0.0091
C(prob, Treatment("high"))[T.opposite]:C(stim, Treatment("bull_lo"))[T.nonface]  -5.3160  2.8272  0.0601
reward                                                                           -4.2173  1.2481  0.0007
reward:C(prob, Treatment("high"))[T.neighbour]                                    4.5869  1.4733  0.0018
reward:C(prob, Treatment("high"))[T.opposite]                                    11.6685  1.9306  0.0000

## pv ~ factors (Feynman, n=2995); coefficients with p < 0.1 (log RT: x100 ~ % change)
                                                                                     coef       se       p
C(prob, Treatment("high"))[T.neighbour]                                           25.1276  11.6485  0.0310
C(stim, Treatment("bull_lo"))[T.face]                                             61.3632  15.8403  0.0001
C(stim, Treatment("bull_lo"))[T.nonface]                                          62.4047  12.9174  0.0000
C(prob, Treatment("high"))[T.neighbour]:C(stim, Treatment("bull_lo"))[T.nonface] -29.9088  14.6904  0.0418
reward                                                                            30.9053  13.5590  0.0226
reward:C(prob, Treatment("high"))[T.neighbour]                                   -30.8133  11.6831  0.0084
reward:C(prob, Treatment("high"))[T.opposite]                                    -42.3303  17.7305  0.0170
reward:C(stim, Treatment("bull_lo"))[T.nonface]                                  -27.1745  12.9500  0.0359

## pv ~ factors (Newton, n=2247); coefficients with p < 0.1 (log RT: x100 ~ % change)
                                                                                      coef       se       p
C(prob, Treatment("high"))[T.neighbour]                                           153.9716  24.3283  0.0000
C(stim, Treatment("bull_lo"))[T.face]                                              85.0895  10.6595  0.0000
C(stim, Treatment("bull_lo"))[T.nonface]                                           98.7287   8.0545  0.0000
C(prob, Treatment("high"))[T.neighbour]:C(stim, Treatment("bull_lo"))[T.bull_hi]   21.9973   6.8372  0.0013
C(prob, Treatment("high"))[T.neighbour]:C(stim, Treatment("bull_lo"))[T.nonface]  -27.2501  10.6911  0.0108
reward:C(stim, Treatment("bull_lo"))[T.face]                                      -20.3456  10.2054  0.0462

## pv ~ factors (both monkeys, n=5242); coefficients with p < 0.1 (log RT: x100 ~ % change)
                                                                                     coef       se       p
C(prob, Treatment("high"))[T.neighbour]                                           80.3367  14.5564  0.0000
C(stim, Treatment("bull_lo"))[T.face]                                             71.4456  10.1761  0.0000
C(stim, Treatment("bull_lo"))[T.nonface]                                          77.9480   8.6323  0.0000
C(prob, Treatment("high"))[T.neighbour]:C(stim, Treatment("bull_lo"))[T.nonface] -29.1876   9.4797  0.0021
reward                                                                            17.6641   8.7565  0.0437
reward:C(prob, Treatment("high"))[T.neighbour]                                   -24.8495   8.4778  0.0034
reward:C(prob, Treatment("high"))[T.opposite]                                    -21.9022  11.7631  0.0626
reward:C(stim, Treatment("bull_lo"))[T.face]                                     -16.6350   9.6908  0.0861
reward:C(stim, Treatment("bull_lo"))[T.nonface]                                  -23.6030   8.9138  0.0081

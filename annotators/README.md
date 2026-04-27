# Annotators V4
- use lemmatized words instead of basic ones
- tweak what keyword LFs are used
- 

## Experiments
### #1 Naive Chat-GPT-based lemmatized keyword search
============================================================
GLOBAL METRICS
============================================================
Coverage (≥1 LF fires): 0.021
Average #LFs per doc: 0.02
Conflict rate (PRO vs CONTRA): 0.000

Label distribution (LF votes):
CONTRA    0.6175
PRO       0.3825

============================================================
LF STATISTICS
============================================================
                        LF  Coverage  Fires  PRO  CONTRA  NEUTRAL  Abstain
                       hmm  0.064486   1171  979     192        0    16988
     lf_con_negative_modal  0.006113    111    0     111        0    18048
               lf_pro_weak  0.005892    107  107       0        0    18052
               lf_con_weak  0.004791     87    0      87        0    18072
        lf_pro_modal_power  0.002093     38   38       0        0    18121
               lf_con_rule  0.001322     24    0      24        0    18135
              lf_con_abuse  0.000826     15    0      15        0    18144
         lf_pro_efficiency  0.000330      6    6       0        0    18153
        lf_con_rule_of_law  0.000275      5    0       5        0    18154
   lf_con_democracy_attack  0.000165      3    0       3        0    18156
            lf_con_erosion  0.000110      2    0       2        0    18157
             lf_pro_crisis  0.000110      2    2       0        0    18157
   lf_pro_judiciary_attack  0.000000      0    0       0        0    18159
          lf_pro_emergency  0.000000      0    0       0        0    18159
  lf_pro_opposition_attack  0.000000      0    0       0        0    18159
          lf_con_overreach  0.000000      0    0       0        0    18159
lf_con_institution_defense  0.000000      0    0       0        0    18159
      lf_pro_strong_leader  0.000000      0    0       0        0    18159
              lf_con_broad  0.000000      0    0       0        0    18159

- welp, these results ain't all that great, to be frank
- coverage is up, but there's a lot more garbage mixed in
- even so, coverage is not very high anyways, there's a bit of conflict (sometimes), but not too much of that, either
- all in all, this calls for a more serious keyword extraction pass
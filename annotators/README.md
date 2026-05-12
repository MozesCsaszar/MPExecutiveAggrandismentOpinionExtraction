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


#### afrer adding discovery-based labels (embeddings), second iteration
Finished E-step with 13985 documents
         4  -12528.81431636     +13.90932396

============================================================
GLOBAL METRICS
============================================================
Coverage (≥1 LF fires): 0.069
Average #LFs per doc: 0.08
Conflict rate (PRO vs CONTRA): 0.003

Label distribution (LF votes):
PRO        0.348274
CONTRA     0.343663
NEUTRAL    0.307849
CON        0.000214

============================================================
LF STATISTICS
============================================================
                          LF  Coverage  Fires  PRO  CONTRA  NEUTRAL  Abstain
                         hmm  0.113356  13973 9002    2196     2775   109294
lf_gemini_flash_v1_annotator  0.021904   2700  135     554     2011   120567
            lf_discovery_pro  0.017685   2180 2180       0        0   121087
         lf_discovery_contra  0.007463    920    0     920        0   122347
        lf_discovery_neutral  0.006977    860    0       0      860   122407
       lf_con_negative_modal  0.006936    855    0     855        0   122412
                 lf_pro_weak  0.004932    608  608       0        0   122659
                 lf_con_weak  0.004121    508    0     508        0   122759
          lf_pro_modal_power  0.002247    277  277       0        0   122990
                 lf_con_rule  0.001598    197    0     197        0   123070
                lf_con_abuse  0.000657     81    0      81        0   123186
          lf_con_rule_of_law  0.000462     57    0      57        0   123210
               lf_pro_crisis  0.000195     24   24       0        0   123243
     lf_con_democracy_attack  0.000187     23    0      23        0   123244
           lf_pro_efficiency  0.000178     22   22       0        0   123245
              lf_con_erosion  0.000081     10    0      10        0   123257
                lf_con_broad  0.000016      2    0       0        0   123265
     lf_pro_judiciary_attack  0.000016      2    2       0        0   123265
            lf_pro_emergency  0.000000      0    0       0        0   123267
        lf_pro_strong_leader  0.000000      0    0       0        0   123267
    lf_pro_opposition_attack  0.000000      0    0       0        0   123267
  lf_con_institution_defense  0.000000      0    0       0        0   123267
            lf_con_overreach  0.000000      0    0       0        0   123267


- decent, but a tad bit too many pros and cons
#### training data:
TOTAL CONTRA: 2196, NEUTRAL: 2775, PRO: 9002
SELECTED CONTRA: 2081, NEUTRAL: 2775, PRO: 2081

- decent, but too many pros
- selection is 30%, 40%, 30%

#### training stats:
{'eval_loss': '0.7715', 'eval_accuracy': {'accuracy': 0.6570605187319885}, 'eval_precision': [0.5946969696969697, 0.7604562737642585, 0.592814371257485], 'eval_recall': [0.6946902654867256, 0.7407407407407407, 0.5], 'eval_f1': [0.6408163265306123, 0.7504690431519699, 0.5424657534246575], 'eval_runtime': '9.691', 'eval_samples_per_second': '71.61', 'eval_steps_per_second': '4.54', 'epoch': '1'}
{'loss': '0.8501', 'grad_norm': '15.04', 'learning_rate': '1.149e-05', 'epoch': '1.279'}                                          
{'eval_loss': '0.7146', 'eval_accuracy': {'accuracy': 0.7204610951008645}, 'eval_precision': [0.7393617021276596, 0.7962264150943397, 0.6224066390041494], 'eval_recall': [0.6150442477876106, 0.7814814814814814, 0.7575757575757576], 'eval_f1': [0.6714975845410628, 0.788785046728972, 0.683371298405467], 'eval_runtime': '9.697', 'eval_samples_per_second': '71.56', 'eval_steps_per_second': '4.537', 'epoch': '2'}
{'loss': '0.5768', 'grad_norm': '15.18', 'learning_rate': '2.967e-06', 'epoch': '2.558'}
{'eval_loss': '0.7201', 'eval_accuracy': {'accuracy': 0.7492795389048992}, 'eval_precision': [0.7525252525252525, 0.778169014084507, 0.7075471698113207], 'eval_recall': [0.6592920353982301, 0.8185185185185185, 0.7575757575757576], 'eval_f1': [0.7028301886792453, 0.7978339350180506, 0.7317073170731707], 'eval_runtime'
: '9.661', 'eval_samples_per_second': '71.83', 'eval_steps_per_second': '4.554', 'epoch': '3'}
{'train_runtime': '1032', 'train_samples_per_second': '18.15', 'train_steps_per_second': '1.137', 'train_loss': '0.6785', 'epoch': '3'}

- overall, it seems decent, so there's learnability in there for sure.
- maybe 1-2 more epochs? (though that might overfit)
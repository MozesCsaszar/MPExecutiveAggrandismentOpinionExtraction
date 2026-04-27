# Annotators V3
- contains a more sophisticated keyword-based matching approach

## First try
Finished E-step with 111107 documents
         4  -17553.48880634      +0.05555386

============================================================
GLOBAL METRICS
============================================================
Coverage (≥1 LF fires): 0.869
Average #LFs per doc: 1.01
Conflict rate (PRO vs CONTRA): 0.000

Label distribution (LF votes):
NEUTRAL    0.998628
CONTRA     0.000810
PRO        0.000562

============================================================
LF STATISTICS
============================================================
                            LF  Coverage  Fires  PRO  CONTRA  NEUTRAL  Abstain
                           hmm  0.901279 111098 4019      18   107061    12169
        lf_neutral_no_politics  0.867402 106922    0       0   106922    16345
              lf_neutral_short  0.124080  15295    0       0    15295   107972
         lf_neutral_procedural  0.017831   2198    0       0     2198   121069
            lf_con_rule_of_law  0.000576     71    0      71        0   123196
       lf_pro_judiciary_attack  0.000381     47   47       0        0   123220
              lf_neutral_quote  0.000211     26    0       0       26   123241
       lf_con_democracy_attack  0.000187     23    0      23        0   123244
                   lf_pro_weak  0.000146     18   18       0        0   123249
        lf_neutral_descriptive  0.000089     11    0       0       11   123256
              lf_con_overreach  0.000057      7    0       7        0   123260
          lf_pro_strong_leader  0.000024      3    3       0        0   123264
              lf_pro_emergency  0.000016      2    2       0        0   123265
    lf_con_institution_defense  0.000000      0    0       0        0   123267
      lf_pro_opposition_attack  0.000000      0    0       0        0   123267

============================================================
EXAMPLES
============================================================

HIGH_CONFLICT:
 - [[u2017-03-06-305]] Persze, ismerjük az ilyen folyamatok dinamizmusát, valószínűleg nem állnak meg addig, amíg a teljes piacra rá nem teszik a kezüket.Ezzel lassan bezárul a kör: kézi irányítás és teljes torz átalakítás a közmédiumokon, ezt tapasztaljuk főleg a Magyar 1-nél, a HírTV2-vé alakított és teljesen a kormány szócsövévé, torz szócsövévé alakított Magyar 1-nél, vagy nézhetjük akár a Vajna-Habony-párossal fémjelzett országos kereskedelmi és hírcsatornákat és a hírportálokat, valamint Mészáros Lőrinccel az országos és megyei lapokat, amik hasonló sorsra jutottak.Persze, látjuk azt is, kicsit még az országos sajtóról beszélve, hogy ezen két úriember fennhatósága alatt lévő kereskedelmi médiumok hogyan telepednek rá a vidéki emberek életére is, hogyan próbálják nagyon sokszor diktatórikus eszközökkel elérni azt, hogy a műsorszolgáltatók szinte más csatornákat már ne is vehessenek be a repertoárjukba és a választható csatornák közé, esetleg csak nagyon drága csomagokba, mint azokban a médiumokban, ahol ennek a két, idézőjelben vett úriembernek a fennhatósága tapasztalható.Persze, én megértem az e folyamatok mögött lévő motivációt.
 - [[u2017-03-07-106]] Ugyanis az alkotmányozó hatalomnak is egészen addig tiszteletben kell tartania az alkotmány rendelkezéseit, ameddig azok hatályban vannak, márpedig az alkotmány, a korábbi alkotmány áttételesen, a mostani Alaptörvény egészen konkrétan tartalmazza azt, hogy a jogállamiságnak az egyik pillére a hatalommegosztás és a fékek és ellensúlyok rendszere.
 - [[u2017-03-07-106]] Márpedig hogyha azt teszi, hogy a neki nem tetsző alkotmánybírósági döntéseket gyakorlatilag alkotmányozás útján egyszerűen felülírja, akkor nem él a hatalmával, hanem visszaél azzal, és gyakorlatilag kiiktat a fékek és ellensúlyok rendszeréből egy független állami szervet, ez pedig nem felel meg sem a régi alkotmány rendelkezéseinek, sem az új Alaptörvény rendelkezéseinek.
 - [[u2017-03-07-106]] Ezekről írtak ezek a szerzők a tanulmányokban, amit én most rendkívül egyszerű nyelvezettel foglalok össze; egyébként magam is írtam ilyen tanulmányt, de ez teljesen mindegy, mert én egy elfogult politikus vagyok itt a Házban, ezzel szemben viszont nem nagyon olvastam ennek a cáfolatát.  Én tehát azt gondolom, tisztelt képviselőtársaim, hogy az alkotmányozó hatalom sem helyezheti magát kívül az alkotmány rendelkezésein, a jogállam általános kritériumrendszerén, és főleg nem a jogállamiság eszméjén, márpedig akkor ezt tette. És hogy ne ez lett volna a helyzet, tisztelt képviselőtársam?
 - [[u2017-03-08-7]] Az Alkotmánybíróság következetes álláspontja szerint az állam társadalommal szembeni alkotmányos kötelezettsége a büntetőigény késedelem nélküli érvényesítése, ami a jogállamiság normatív tartalmából és a tisztességes eljáráshoz való alkotmányos jogból vezethető le.

PRO_EXAMPLES:
 - [[u2017-02-20-1]] Másodszor: hatályba lépett a 2017. év költségvetése, a kormány az Országgyűlés döntésének megfelelően elkezdte annak végrehajtását.
 - [[u2017-02-20-1]] A 2010-es kormányváltás óta több mint 700 ezerrel nőtt a foglalkoztatottak száma, ami döntően a hazai versenyszférában következett be, ahol 486 ezer fővel nőtt a munkában állók száma, időarányosan tehát jól állunk a tíz évre vállalt egymillió új munkahelyet illetően.
 - [[u2017-02-20-1]] A kormány öt olyan veszélyforrást nevesített, amellyel 2017-ben szembe kell néznünk.
 - [[u2017-02-20-1]] A kérelmet benyújtó migránsokat ügyük jogerős elbírálásáig a kormány javaslata szerint őrizetben kell tartani, szemben a mostani helyzettel, amikor szabadon mozoghatnak, nemcsak Magyarország területén, hanem az egész schengeni övezetben.
 - [[u2017-02-20-3]] Miniszterelnök úr, önnek egy éve van, egy éve van, kezdje el vágni a centit, és készüljön fel arra, hogy személyesen kell felelnie Pharaonért, Mészáros Lőrincért, a kötvényért és mindenféle piszkos ügyéért.

CON_EXAMPLES:
 - [[u2017-03-13-5]] Egy normális köztársaságban függetlenek a bíróságok, független az ügyészség, függetlenül működik az Alkotmánybíróság, szakmai alapon, független a Nemzeti Bank, működik a fékek és ellensúlyok rendszere.
 - [[u2017-04-19-91]] Látom, hogy Hende képviselő úr, csakúgy, mint a többi kormánypárti képviselő, rendkívüli módon tisztában van azzal, hogy melyik rendszernek mik a sajátosságai; akkor biztos, hogy ha más nem, akkor otthon maguknak bevallják, hogy egyébként ma Magyarországon egy autoriter rendszer működik, ez csak a demokrácia álcája.
 - [[u2017-04-19-191]] A nemzetbiztonsági célú titkos információgyűjtés alkotmányos kontrolljának koncepcionális újragondolását nem úszhatja meg ugyanis a kormány.
 - [[u2017-04-24-193]] A kormány évek óta a szemünk láttára, lépésről lépésre építi ki Magyarországon az autoriter rendszert.
 - [[u2017-05-08-5]] Ez az alacsony, köpcös ember hatalomra kerülve felszámolta a jogállamiságot, felszámolta a sajtószabadságot, tökélyre fejlesztette a kormányzati propagandát, diktatúrát épített ki, majd belesodorta Európát egy korábban soha nem látott pusztító háborúba.

NEUTRAL_EXAMPLES:
 - [[u2017-02-20-1]] Tisztelt Elnök Úr!
 - [[u2017-02-20-1]] Tisztelt Képviselőtársaim!
 - [[u2017-02-20-1]] Tisztelt Ház!
 - [[u2017-02-20-1]] Ma négy dologról kell beszélnem.
 - [[u2017-02-20-1]] Tisztelt Ház!


 ### Experiment 2

 Finished E-step with 23738 documents
         4  -14726.00739620      +0.41344886
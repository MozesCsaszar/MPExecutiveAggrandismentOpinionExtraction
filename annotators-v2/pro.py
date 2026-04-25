from spacy.tokens import Doc
import skweak
from .helpers import _lf_preprocess


# Strong leadership / centralisation framing [[ChatGPT]]
PRO_STRONG = [
    "erős kormány",
    "erős vezetés",
    "határozott kormányzás",
    "cselekvőképes kormány",
    "gyors döntéshozatal",
    "nemzeti érdek elsőbbsége",
]


def lf_pro_strong_leader(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in PRO_STRONG):
        yield doc[0].i, doc[-1].i + 1, "PRO"


lf_pro_strong_leader = skweak.heuristics.FunctionAnnotator(
    "lf_pro_strong_leader", lf_pro_strong_leader
)  # type: ignore

# Emergency powers justification [[ChatGPT]]
PRO_EMERGENCY = [
    "rendkívüli állapot szükséges",
    "veszélyhelyzet indokolja",
    "felhatalmazás szükséges",
    "különleges jogrend indokolt",
]


def lf_pro_emergency(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in PRO_EMERGENCY):
        yield doc[0].i, doc[-1].i + 1, "PRO"


lf_pro_emergency = skweak.heuristics.FunctionAnnotator(
    "lf_pro_emergency", lf_pro_emergency
)  # type: ignore

# Judiciary criticism [[ChatGPT]]
JUDICIARY = ["bíróság", "alkotmánybíróság", "bírók"]
NEG = ["akadályoz", "lassú", "politikai", "elfogult"]


def lf_pro_judiciary_attack(doc: Doc):
    text = _lf_preprocess(doc)
    if any(j in text for j in JUDICIARY) and any(n in text for n in NEG):
        yield doc[0].i, doc[-1].i + 1, "PRO"


lf_pro_judiciary_attack = skweak.heuristics.FunctionAnnotator(
    "lf_pro_judiciary_attack", lf_pro_judiciary_attack
)  # type: ignore

# Anti-opposition framing (weak proxy) [[ChatGPT]]
PRO_OPPOSITION_ATTACK = [
    "ellenzék akadályozza",
    "ellenzék hátráltatja",
    "ellenzék felelőtlen",
]


def lf_pro_opposition_attack(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in PRO_OPPOSITION_ATTACK):
        yield doc[0].i, doc[-1].i + 1, "PRO"


lf_pro_opposition_attack = skweak.heuristics.FunctionAnnotator(
    "lf_pro_opposition_attack", lf_pro_opposition_attack
)  # type: ignore


# Government party prior (from ParlaMint) [[ChatGPT]]
# TODO: Make this work
def lf_pro_government_party(doc):
    party = doc._.party  # from ParlaMint metadata
    gov_parties = doc._.gov_parties
    if party in gov_parties:
        return "PRO"


pro_annotator = skweak.base.CombinedAnnotator()
pro_annotator.add_annotators(
    lf_pro_emergency,  # type: ignore
    lf_pro_opposition_attack,  # type: ignore
    lf_pro_judiciary_attack,  # type: ignore
    lf_pro_strong_leader,  # type: ignore
)

from spacy.tokens import Doc
import skweak
from .helpers import _lf_preprocess
from .consts import NEGATIVE, PRO_ACTION, PRO_JUSTIFICATION, PRO_POWER, INSTITUTIONS
from .lf_generators import generate_variants, window_match


# Strong leadership / centralisation framing [[ChatGPT]]
_PRO_STRONG = [
    "erős kormány",
    "erős vezetés",
    "határozott kormányzás",
    "cselekvőképes kormány",
    "gyors döntéshozatal",
    "nemzeti érdek elsőbbsége",
]


def lf_pro_strong_leader(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in _PRO_STRONG):
        yield doc[0].i, doc[-1].i + 1, "PRO"


lf_pro_strong_leader = skweak.heuristics.FunctionAnnotator(
    "lf_pro_strong_leader", lf_pro_strong_leader
)  # type: ignore

# Emergency powers justification [[ChatGPT]]
_PRO_EMERGENCY = [
    "rendkívüli állapot szükséges",
    "veszélyhelyzet indokolja",
    "felhatalmazás szükséges",
    "különleges jogrend indokolt",
]


def lf_pro_emergency(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in _PRO_EMERGENCY):
        yield doc[0].i, doc[-1].i + 1, "PRO"


lf_pro_emergency = skweak.heuristics.FunctionAnnotator(
    "lf_pro_emergency", lf_pro_emergency
)  # type: ignore

# Judiciary criticism [[ChatGPT]]
_JUDICIARY = ["bíróság", "alkotmánybíróság", "bírók"]


def lf_pro_judiciary_attack(doc: Doc):
    text = _lf_preprocess(doc)
    if any(j in text for j in _JUDICIARY) and any(n in text for n in NEGATIVE):
        yield doc[0].i, doc[-1].i + 1, "PRO"


lf_pro_judiciary_attack = skweak.heuristics.FunctionAnnotator(
    "lf_pro_judiciary_attack", lf_pro_judiciary_attack
)  # type: ignore

# Anti-opposition framing (weak proxy) [[ChatGPT]]
_PRO_OPPOSITION_ATTACK = [
    "ellenzék akadályozza",
    "ellenzék hátráltatja",
    "ellenzék felelőtlen",
]


def lf_pro_opposition_attack(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in _PRO_OPPOSITION_ATTACK):
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


lfs_pro_action = generate_variants("pro_action", "PRO", PRO_POWER, PRO_ACTION)

lfs_pro_justification = generate_variants(
    "pro_justification", "PRO", PRO_POWER, PRO_JUSTIFICATION
)

lfs_pro_judiciary = generate_variants(
    "pro_judiciary_attack", "PRO", INSTITUTIONS, NEGATIVE
)


def lf_pro_broad(doc: Doc):
    window_matched = window_match(doc, PRO_POWER, PRO_ACTION, 10)
    if window_matched:
        yield doc[0].i, doc[-1].i + 1, "PRO"


lf_pro_broad = skweak.heuristics.FunctionAnnotator(
    "lf_pro_weak", lf_pro_broad
)  # type: ignore

pro_annotator = skweak.base.CombinedAnnotator()
pro_annotator.add_annotators(
    lf_pro_emergency,  # type: ignore
    lf_pro_opposition_attack,  # type: ignore
    lf_pro_judiciary_attack,  # type: ignore
    lf_pro_strong_leader,  # type: ignore
    lf_pro_broad,  # type: ignore
    *lfs_pro_action,
    *lfs_pro_justification,
    *lfs_pro_judiciary
)

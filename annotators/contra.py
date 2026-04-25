from spacy.tokens import Doc
import skweak
from .helpers import _lf_preprocess
from .consts import CON_DEMO, CON_NEGATIVE, CON_RULE_OF_LAW
from .lf_generators import generate_variants

# Rule of law / checks and balances [[ChatGPT]]
_CON_RULE_OF_LAW = [
    "jogállamiság",
    "fékek és ellensúlyok",
    "hatalmi ágak szétválasztása",
    "alkotmányos kontroll",
]


def lf_con_rule_of_law(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in _CON_RULE_OF_LAW):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_rule_of_law = skweak.heuristics.FunctionAnnotator(
    "lf_con_rule_of_law", lf_con_rule_of_law
)  # type: ignore

# Democracy violation accusations [[ChatGPT]]+
_CON_DEMOCRACY_ATTACK = [
    "autoriter",
    "diktatórikus",
    "hatalomkoncentráció",
]


def lf_con_democracy_attack(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in _CON_DEMOCRACY_ATTACK):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_democracy_attack = skweak.heuristics.FunctionAnnotator(
    "lf_con_democracy_attack", lf_con_democracy_attack
)  # type: ignore


# Defense of judiciary / institutions [[ChatGPT]]
def lf_con_institution_defense(doc: Doc):
    text = _lf_preprocess(doc)
    if "védeni kell" in text or "meg kell védeni" in text:
        if "bíróság" in text or "alkotmány" in text:
            yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_institution_defense = skweak.heuristics.FunctionAnnotator(
    "lf_con_institution_defense", lf_con_institution_defense
)  # type: ignore


# Government overreach framing [[ChatGPT]]
_CON_OVERREACH = ["kormány túlkapás", "hatalommal való visszaélés", "önkényes döntés"]


def lf_con_overreach(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in _CON_OVERREACH):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_overreach = skweak.heuristics.FunctionAnnotator(
    "lf_con_overreach", lf_con_overreach
)  # type: ignore


#  Opposition party prior [[ChatGPT]]
# TODO: Make this work
def lf_con_opposition_party(doc):
    party = doc._.party
    gov_parties = doc._.gov_parties
    if party not in gov_parties:
        return "CON"


lfs_con_democracy = generate_variants("con_democracy", "CON", CON_DEMO, CON_NEGATIVE)

lfs_con_rule = generate_variants(
    "con_rule_of_law", "CON", CON_RULE_OF_LAW, CON_NEGATIVE
)


contra_annotator = skweak.base.CombinedAnnotator()
contra_annotator.add_annotators(
    lf_con_democracy_attack,  # type: ignore
    lf_con_institution_defense,  # type: ignore
    lf_con_overreach,  # type: ignore
    lf_con_rule_of_law,  # type: ignore
    *lfs_con_democracy,
    *lfs_con_rule
)

from spacy.tokens import Doc
import skweak
from .consts import (
    CON_ABUSE,
    CON_DEMO,
    CON_EROSION,
    CON_NEGATIVE,
    CON_RULE,
    CON_RULE_OF_LAW,
    MODAL_NEG,
    TARGET_POWER,
)
from .lf_generators import (
    generate_variants,
    window_match_lemma,
    contains_any,
    get_lemmas,
)

# Rule of law / checks and balances [[ChatGPT]]
_CON_RULE_OF_LAW = [
    "jogállamiság",
    "fékek és ellensúlyok",
    "hatalmi ágak szétválasztása",
    "alkotmányos kontroll",
]


def lf_con_rule_of_law(doc: Doc):
    if contains_any(doc, _CON_RULE_OF_LAW):
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
    if contains_any(doc, _CON_DEMOCRACY_ATTACK):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_democracy_attack = skweak.heuristics.FunctionAnnotator(
    "lf_con_democracy_attack", lf_con_democracy_attack
)  # type: ignore


# Defense of judiciary / institutions [[ChatGPT]]
def lf_con_institution_defense(doc: Doc):
    text = get_lemmas(doc)
    if "védeni kell" in text or "meg kell védeni" in text:
        if "bíróság" in text or "alkotmány" in text:
            yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_institution_defense = skweak.heuristics.FunctionAnnotator(
    "lf_con_institution_defense", lf_con_institution_defense
)  # type: ignore


# Government overreach framing [[ChatGPT]]
_CON_OVERREACH = ["kormány túlkapás", "hatalommal való visszaélés", "önkényes döntés"]


def lf_con_overreach(doc: Doc):
    if contains_any(doc, _CON_OVERREACH):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_overreach = skweak.heuristics.FunctionAnnotator(
    "lf_con_overreach", lf_con_overreach
)  # type: ignore


#  Opposition party prior [[ChatGPT]]
# TODO: Make this work
def lf_con_opposition_party(doc):
    if hasattr(doc._, "party") and hasattr(doc._, "gov_parties"):
        if doc._.party not in doc._.gov_parties:
            yield doc[0].i, doc[-1].i + 1, "CONTRA"


lfs_con_democracy = generate_variants("con_democracy", "CON", CON_DEMO, CON_NEGATIVE)

lfs_con_rule = generate_variants(
    "con_rule_of_law", "CON", CON_RULE_OF_LAW, CON_NEGATIVE
)


def lf_con_broad(doc: Doc):
    if window_match_lemma(doc, CON_NEGATIVE, CON_DEMO, 10):
        yield doc[0].i, doc[-1].i, ("CON")


lf_con_broad = skweak.heuristics.FunctionAnnotator(
    "lf_con_broad", lf_con_broad
)  # type: ignore


def lf_con_rule(doc):
    if contains_any(doc, CON_RULE):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_rule = skweak.heuristics.FunctionAnnotator(
    "lf_con_rule", lf_con_rule
)  # type: ignore


def lf_con_erosion(doc):
    if window_match_lemma(doc, TARGET_POWER, CON_EROSION, 5):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_erosion = skweak.heuristics.FunctionAnnotator(
    "lf_con_erosion", lf_con_erosion
)  # type: ignore


def lf_con_abuse(doc):
    if contains_any(doc, CON_ABUSE):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_abuse = skweak.heuristics.FunctionAnnotator(
    "lf_con_abuse", lf_con_abuse
)  # type: ignore


def lf_con_negative_modal(doc):
    if window_match_lemma(doc, TARGET_POWER, MODAL_NEG, 5):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_negative_modal = skweak.heuristics.FunctionAnnotator(
    "lf_con_negative_modal", lf_con_negative_modal
)  # type: ignore


def lf_con_weak(doc):
    if contains_any(doc, CON_EROSION):
        yield doc[0].i, doc[-1].i + 1, "CONTRA"


lf_con_weak = skweak.heuristics.FunctionAnnotator(
    "lf_con_weak", lf_con_weak
)  # type: ignore


contra_annotator = skweak.base.CombinedAnnotator()
contra_annotator.add_annotators(
    lf_con_democracy_attack,  # type: ignore
    lf_con_institution_defense,  # type: ignore
    lf_con_overreach,  # type: ignore
    lf_con_rule_of_law,  # type: ignore
    lf_con_broad,  # type: ignore
    lf_con_rule,  # type: ignore
    lf_con_erosion,  # type: ignore
    lf_con_abuse,  # type: ignore
    lf_con_negative_modal,  # type: ignore
    lf_con_weak,  # type: ignore
    *lfs_con_democracy,
    *lfs_con_rule
)

from spacy.tokens import Doc
import skweak
from .helpers import _lf_preprocess

# Procedural speech [[ChatGPT]]
NEUTRAL_PROCEDURAL = [
    "napirendi pont",
    "köszönöm a szót",
    "szavazás következik",
    "jegyzőkönyv szerint",
    "ülést megnyitom",
]


def lf_neutral_procedural(doc: Doc):
    text = _lf_preprocess(doc)
    if any(p in text for p in NEUTRAL_PROCEDURAL):
        yield doc[0].i, doc[-1].i + 1, "NEUTRAL"


lf_neutral_procedural = skweak.heuristics.FunctionAnnotator(
    "lf_neutral_procedural", lf_neutral_procedural
)  # type: ignore


# Administrative / descriptive [[ChatGPT]]
def lf_neutral_descriptive(doc: Doc):
    text = _lf_preprocess(doc)
    if "jelentés szerint" in text or "statisztika alapján" in text:
        yield doc[0].i, doc[-1].i + 1, "NEUTRAL"


lf_neutral_descriptive = skweak.heuristics.FunctionAnnotator(
    "lf_neutral_descriptive", lf_neutral_descriptive
)  # type: ignore


# No political actors mentioned [[ChatGPT]]
# NOTE: I'm not convinced about this one
POLITICAL_TERMS = ["kormány", "miniszterelnök", "bíróság", "alkotmány", "ellenzék"]


def lf_neutral_no_politics(doc: Doc):
    text = _lf_preprocess(doc)
    if not any(p in text for p in POLITICAL_TERMS):
        yield doc[0].i, doc[-1].i + 1, "NEUTRAL"


lf_neutral_no_politics = skweak.heuristics.FunctionAnnotator(
    "lf_neutral_no_politics", lf_neutral_no_politics
)  # type: ignore

# Very short utterances [[ChatGPT]]
MIN_WORD_COUNT = 5


def lf_neutral_short(doc: Doc):
    if len(doc) < MIN_WORD_COUNT:
        yield doc[0].i, doc[-1].i + 1, "NEUTRAL"


lf_neutral_short = skweak.heuristics.FunctionAnnotator(
    "lf_neutral_short", lf_neutral_short
)  # type: ignore


# Quote / attribution detection [[ChatGPT]]
def lf_neutral_quote(doc: Doc):
    text = _lf_preprocess(doc)
    if "az ellenzék szerint" in text or "idézi" in text:
        yield doc[0].i, doc[-1].i + 1, "NEUTRAL"


lf_neutral_quote = skweak.heuristics.FunctionAnnotator(
    "lf_neutral_quote", lf_neutral_quote
)  # type: ignore


neutral_annotator = skweak.base.CombinedAnnotator()
neutral_annotator.add_annotators(
    lf_neutral_descriptive,  # type: ignore
    lf_neutral_no_politics,  # type: ignore
    lf_neutral_procedural,  # type: ignore
    lf_neutral_quote,  # type: ignore
    lf_neutral_short,  # type: ignore
)

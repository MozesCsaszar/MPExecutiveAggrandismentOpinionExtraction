import skweak
from spacy.tokens import Doc


# [[ChatGPT]]+
def get_lemmas(doc: Doc):
    return [t.lemma_.lower() for t in doc]


# [[ChatGPT]]+
def contains_any(doc: Doc, lexicon: list[str]):
    lemmas = get_lemmas(doc)
    return any(w in lemmas for w in lexicon)


# [[ChatGPT]]+
def window_match_lemma(doc: Doc, words1: list[str], words2: list[str], window=5):
    lemmas = get_lemmas(doc)
    for i, t in enumerate(lemmas):
        if t in words1:
            for j in range(max(0, i - window), min(len(lemmas), i + window)):
                if lemmas[j] in words2:
                    return True
    return False


# [[ChatGPT]]+
def has_negation(doc: Doc, target_words: list[str], window=3):
    lemmas = get_lemmas(doc)
    for i, t in enumerate(lemmas):
        if t in target_words:
            for j in range(max(0, i - window), min(len(lemmas), i + window)):
                if lemmas[j] == "nem":
                    return True
    return False


# [[ChatGPT]]+
def make_lf(name, label, condition_fn):
    def lf(doc: Doc):
        if condition_fn(doc):
            yield doc[0].i, doc[-1].i + 1, label

    lf.__name__ = name
    return skweak.heuristics.FunctionAnnotator(name, lf)  # type: ignore


# [[ChatGPT]]
def generate_variants(base_name, label, concept1, concept2):
    lfs = []

    # Variant 1: simple co-occurrence
    def cond1(doc):
        text = doc.text.lower()
        return any(w1 in text for w1 in concept1) and any(w2 in text for w2 in concept2)

    lfs.append(make_lf(base_name + "_cooc", label, cond1))

    # Variant 2: window match
    def cond2(doc):
        return window_match_lemma(doc, concept1, concept2, window=5)

    lfs.append(make_lf(base_name + "_window", label, cond2))

    # Variant 3: larger window (looser)
    def cond3(doc):
        return window_match_lemma(doc, concept1, concept2, window=10)

    lfs.append(make_lf(base_name + "_window_loose", label, cond3))

    return lfs

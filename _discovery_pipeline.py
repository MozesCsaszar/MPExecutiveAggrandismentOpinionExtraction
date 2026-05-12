# ============================================================
# HUNGARIAN POLITICAL SPEECH ANALYSIS PIPELINE
# ============================================================
#
# PURPOSE:
# Detect rhetorical patterns associated with:
# - political aggrandizement
# - authoritarian framing
# - sovereignty rhetoric
# - institutional delegitimization
#
# DESIGN PRINCIPLES:
# - selective lemmatization
# - preserve Hungarian verbal semantics
# - preserve proper nouns
# - phrase extraction
# - dependency extraction
# - domain normalization
#
# ============================================================

# =========================
# INSTALL
# =========================

# pip install huspacy gensim pandas scikit-learn tqdm
# python -m huspacy download hu_core_news_lg


# =========================
# IMPORTS
# =========================

from collections import Counter

import pandas as pd
from tqdm import tqdm

from gensim.models.phrases import Phrases, Phraser

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokens import Doc
import spacy

from utilities import load_docs

# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
# You WILL want to tune these later.
# Especially after inspecting real output.
# ------------------------------------------------------------

ALLOWED_POS = {"NOUN", "PROPN", "VERB", "ADJ"}

REMOVE_POS = {"AUX", "DET", "CCONJ", "SCONJ", "PART", "INTJ", "PUNCT", "SYM", "NUM"}

# ------------------------------------------------------------
# DOMAIN NORMALIZATION
# ------------------------------------------------------------
# Purpose:
# merge semantically equivalent institutional variants
# and rhetorical variants.
#
# Example:
# - bírósági -> bíróság
# - közigazgatási -> közigazgatás
#
# IMPORTANT:
# Keep this SMALL and high precision.
#
# ------------------------------------------------------------

NORMALIZE = {
    "bírósági": "bíróság",
    "közigazgatási": "közigazgatás",
    "kormányzati": "kormány",
    "nemzeti": "nemzet",
    "alkotmányos": "alkotmány",
    "szuverén": "szuverenitás",
}

# TODO: Add stopwords
CUSTOM_STOPWORDS = {
    "hogy",
    "amely",
    "ahol",
    "csak",
    "minden",
    "egyébként",
    "ugye",
    "tehát",
}


# selective normalization
def normalize_token(token):
    """
    Core normalization logic.

    KEY DESIGN CHOICE:
    - nouns -> lemma
    - verbs -> surface form
    - adjectives -> surface form
    - proper nouns -> preserve case
    """

    # Remove stopwords
    if token.is_stop:
        return None

    if token.text.lower() in CUSTOM_STOPWORDS:
        return None

    # Remove punctuation etc.
    if token.pos_ in REMOVE_POS:
        return None

    if token.pos_ not in ALLOWED_POS:
        return None

    # preserve origginal casing for proper nouns
    if token.pos_ == "PROPN":
        form = token.text

    # preserve verb prefixes (so no lemmatization)
    elif token.pos_ == "VERB":
        form = token.text.lower()

    # presever adjective lemmatization
    elif token.pos_ == "ADJ":
        form = token.text.lower()

    # lemmatize nouns
    else:
        form = token.lemma_.lower()

    # apply domain normalization
    form = NORMALIZE.get(form, form)

    # Remove tiny fragments
    if len(form) < 2:
        return None

    return form


# preprocess a single document
def process_document(doc: Doc):
    tokens = []

    for token in doc:
        norm = normalize_token(token)

        if norm:
            tokens.append(norm)

    return tokens


# preprocess docs
def process_corpus(docs: list[Doc]):
    processed = []

    print(f"Processing `{len(docs)}` docs...")

    for doc in tqdm(docs):
        tokens = process_document(doc)

        processed.append(tokens)

    return processed


# phrase detection
def build_phrase_model(processed_texts):
    # TODO: Tune manually by increasing min_count for larger corpora.
    # threshold:
    # higher = stricter phrase acceptance

    phrases = Phrases(processed_texts, min_count=5, threshold=10)
    phraser = Phraser(phrases)

    return phraser


def apply_phrases(processed_texts, phraser):
    return [phraser[text] for text in processed_texts]


# Count token frequencies
def token_frequencies(processed_texts, top_n=100):
    counter = Counter()

    for text in processed_texts:
        counter.update(text)

    return counter.most_common(top_n)


# extract dependencies
def extract_svo(doc):
    """
    Extract subject-verb-object triples.

    VERY IMPORTANT:
    This often gives stronger political signal
    than bag-of-words.
    """

    triples = []

    for token in doc:
        # Main verbs only
        if token.pos_ != "VERB":
            continue

        subject = None
        obj = None

        for child in token.children:
            if child.dep_ in ("nsubj", "csubj"):
                subject = normalize_token(child)
            elif child.dep_ in ("obj", "iobj"):
                obj = normalize_token(child)

        if subject and obj:
            triples.append((subject, token.text.lower(), obj))

    return triples


# extract all SVO triplets
def corpus_svo(docs):
    all_triples = []

    for doc in docs:
        triples = extract_svo(doc)

        all_triples.extend(triples)

    return all_triples


# TF-IDF processing
def build_tfidf(processed_texts):
    joined = [" ".join(tokens) for tokens in processed_texts]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))

    X = vectorizer.fit_transform(joined)

    return X, vectorizer


# Keyness/distinctive terms
def class_keywords(processed_texts, labels, top_n=50):
    """
    labels:
    1 = aggrandizing
    0 = non-aggrandizing
    """

    joined = [" ".join(tokens) for tokens in processed_texts]

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)

    X = vectorizer.fit_transform(joined)

    feature_names = vectorizer.get_feature_names_out()

    df = pd.DataFrame(X.toarray(), columns=feature_names)

    df["label"] = labels

    positive = df[df["label"] == 1]
    negative = df[df["label"] == 0]

    scores = []

    for col in feature_names:
        pos_mean = positive[col].mean()
        neg_mean = negative[col].mean()

        diff = pos_mean - neg_mean

        scores.append((col, diff))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores[:top_n]


if __name__ == "__main__":
    print("Loading model...")
    nlp = spacy.load(
        "hu_core_news_md",
    )

    print("Loading docs...")
    docs = load_docs(nlp, ["2017"], suffix="md") or []

    print("Preprocessing corupus...")
    processed = process_corpus(docs)

    print("Building phrases...")
    phraser = build_phrase_model(processed)
    phrased = apply_phrases(processed, phraser)

    # Counting token frequencies
    print("Counting frequencies...")
    freqs = token_frequencies(phrased)

    print("\nFREQUENCIES:")
    print(freqs[:20])

    # SVO triplets
    print("Constructing triplets...")
    triples = corpus_svo(docs)

    print("\nSVO:")
    print(triples[:3])

    # Keywords
    # print("Finding class keywords...")
    # keywords = class_keywords(phrased, labels)

    # print("\nDISTINCTIVE:")
    # print(keywords[:20])

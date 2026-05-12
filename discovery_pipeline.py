# ============================================================
# EXECUTIVE AGGRANDIZEMENT DISCOVERY PIPELINE
# ============================================================
#
# Find suspicious sentences that might include EA, without using labeled data.
#
# ============================================================
# WHAT CHANGED FROM THE PREVIOUS PIPELINE?
# ============================================================
#
# PIPELINE:
# - sentence-level retrieval
# - embedding similarity
# - prototype matching
# - ranking suspicious sentences
# - clustering rhetorical patterns
#
# Idea:
# - define prototype rhetoric,
# - retrieve semantically similar sentences,
# - review only top-ranked outputs.
#
# ============================================================

import re
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer

from bertopic import BERTopic

import spacy
from spacy.tokens import Doc

from utilities import load_docs, save_docs

# ============================================================
# LOAD HUNGARIAN NLP MODEL
# ============================================================

nlp = spacy.load(
    "hu_core_news_md",
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================
#
# WHY THIS MODEL?
#
# multilingual MiniLM:
# - works well on Hungarian
# - lightweight
# - surprisingly good semantic retrieval
#
# ALTERNATIVES:
# - LaBSE (slower but stronger)
# - multilingual-e5-large
#
# ============================================================

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# Domain Normalization

NORMALIZE = {
    "közigazgatási": "közigazgatás",
    "bírósági": "bíróság",
    "alkotmányos": "alkotmány",
    "nemzeti": "nemzet",
    "hazafias": "nemzet",
    "szuverén": "szuverenitás",
    "független": "szuverenitás",
}


# CUSTOM STOPWORDS

CUSTOM_STOPWORDS = {
    "hogy",
    "amely",
    "ahol",
    "csak",
    "minden",
    "tehát",
    "ugye",
}


# ============================================================
# SELECTIVE NORMALIZATION
# ============================================================
#
# SAME CORE STRATEGY AS BEFORE:
#
# - nouns -> lemma
# - verbs -> surface form
# - adjectives -> surface form
# - proper nouns -> preserve case
#
# WHY?
#
# Hungarian verbal prefixes contain political meaning.
#
# Example:
# - megvédi
# - megkerüli
# - beavatkozik
#
# Pure lemmatization destroys signal.
#
# ============================================================


def normalize_token(token):
    if token.is_stop:
        return None

    if token.text.lower() in CUSTOM_STOPWORDS:
        return None

    if token.is_punct:
        return None

    if token.pos_ in {"DET", "CCONJ", "SCONJ", "AUX"}:
        return None

    # -------------------------
    # PROPER NOUNS
    # -------------------------

    if token.pos_ == "PROPN":
        form = token.text

    # -------------------------
    # VERBS
    # -------------------------

    elif token.pos_ == "VERB":
        form = token.text.lower()

    # -------------------------
    # ADJECTIVES
    # -------------------------

    elif token.pos_ == "ADJ":
        form = token.text.lower()

    # -------------------------
    # NOUNS
    # -------------------------

    elif token.pos_ == "NOUN":
        form = token.lemma_.lower()

    else:
        return None

    form = NORMALIZE.get(form, form)

    if len(form) < 2:
        return None

    return form


# ============================================================
# NORMALIZE SENTENCE
# ============================================================
#
# IMPORTANT:
#
# We normalize ONLY for:
# - retrieval
# - similarity
# - phrase discovery
#
# BUT:
# we ALWAYS preserve original sentences
# for interpretation later.
#
# ============================================================


def normalize_doc(doc: Doc | str):
    tokens = []

    if isinstance(doc, str):
        doc = nlp(doc)

    for token in doc:
        norm = normalize_token(token)

        if norm:
            tokens.append(norm)

    return " ".join(tokens)


# ============================================================
# PROCESS ALL SENTENCES
# ============================================================


def preprocess_docs(docs: list[Doc]):
    normalized = []

    for doc in tqdm(docs):
        normalized.append(normalize_doc(doc))

    return normalized


# ============================================================
# PROTOTYPE SENTENCES
# ============================================================
#
# VERY IMPORTANT.
#
# THIS REPLACES MANUAL LABELS.
#
# Instead of:
# - labeled data
#
# We define:
# - rhetorically prototypical statements.
#
# Then retrieve semantically similar sentences.
#
# ============================================================

# TODO: Find better prototypes
PROTOTYPES = [
    # sovereignty framing
    "A kormány megvédi Magyarország szuverenitását.",
    # emergency framing
    "Rendkívüli intézkedések szükségesek.",
    # anti-foreign interference
    "Külföldi érdekek beavatkoznak az ország ügyeibe.",
    # anti-judiciary framing
    "A bíróságok akadályozzák a kormány munkáját.",
    # centralized authority
    "Erős vezetésre van szükség.",
    # anti-pluralist framing
    "Csak a kormány képviseli a nemzet akaratát.",
]


# ============================================================
# NORMALIZE PROTOTYPES
# ============================================================

normalized_prototypes = [normalize_doc(x) for x in PROTOTYPES]


# ============================================================
# EMBEDDING STEP
# ============================================================
#
# THIS IS THE CORE SEMANTIC RETRIEVAL STEP.
#
# We convert:
# - sentences
# - prototypes
#
# into vectors.
#
# Then:
# - retrieve nearest neighbors.
#
# ============================================================


def build_embeddings(texts):
    return embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


# ============================================================
# RETRIEVE SUSPICIOUS SENTENCES
# ============================================================
#
# HOW IT WORKS:
#
# For each sentence:
# - compare against all prototypes
#
# Final score:
# - maximum similarity to any prototype
#
# WHY MAX?
#
# Because:
# - a sentence only needs to match ONE
#   aggrandizement frame strongly.
#
# ============================================================


def retrieve_suspicious_sentences(
    original_sentences: list[Doc], normalized_sentences, top_k=1000
):
    # create embeddings
    sentence_embeddings = build_embeddings(normalized_sentences)
    prototype_embeddings = build_embeddings(normalized_prototypes)

    # get similarity
    similarities = cosine_similarity(sentence_embeddings, prototype_embeddings)

    # find max similarity
    scores = similarities.max(axis=1)

    # build results
    results = []

    for i, score in enumerate(scores):

        results.append(
            {
                "sentence": original_sentences[i],
                "normalized": normalized_sentences[i],
                "score": float(score),
            }
        )

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ============================================================
# TOPIC CLUSTERING
# ============================================================
#
# WHY BERTopic?
#
# It combines:
# - embeddings
# - clustering
# - keyword extraction
#
# This helps discover:
# - rhetorical subtypes
# - recurring ideological themes
#
# WITHOUT LABELS.
#
# ============================================================


def build_topics(normalized_sentences):
    topic_model = BERTopic(
        language="multilingual", calculate_probabilities=False, verbose=True
    )

    # TODO: Figure out whether I need `probs`` or not; replaced with _ for the time being
    topics, _ = topic_model.fit_transform(normalized_sentences)

    return topic_model, topics


def print_topics(topic_model, top_n=20):
    topic_info = topic_model.get_topic_info()

    print(topic_info.head(top_n))


# ============================================================
# KEY PHRASES VIA TF-IDF
# ============================================================
#
# OPTIONAL BUT USEFUL.
#
# Finds:
# - distinctive rhetorical phrases
#
# among suspicious sentences.
#
# ============================================================


def extract_keyphrases(texts, top_n=100):

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)

    X = vectorizer.fit_transform(texts)

    features = vectorizer.get_feature_names_out()

    scores = np.asarray(X.mean(axis=0)).ravel()

    ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)

    return ranked[:top_n]


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------
    #
    # You said:
    # - sentences are already separated.
    #
    # So:
    # sentences = [...]
    #
    # should contain ONE sentence per item.
    #
    # --------------------------------------------------------

    print("Loading docs...")
    docs = load_docs(nlp, ["2017"], suffix="md") or []
    docs = docs[:100]

    print("Saving docs...")
    save_docs(docs, ["2017"], suffix="0-100")

    # normalize sentences
    normalized_sentences = preprocess_docs(docs)

    # retrieve suspicious
    suspicious = retrieve_suspicious_sentences(docs, normalized_sentences, top_k=100)

    # get top results
    print("\nTOP SUSPICIOUS SENTENCES:\n")

    for x in suspicious[:20]:
        print(round(x["score"], 3), "|", x["sentence"])

    # do some topic modeling
    topic_model, topics = build_topics(normalized_sentences)

    # inspect the topics
    print("\nDISCOVERED TOPICS:\n")

    print_topics(topic_model)

    keyphrases = extract_keyphrases(normalized_sentences)

    print("\nKEY PHRASES:\n")

    for phrase, score in keyphrases[:50]:

        print(round(score, 5), phrase)

import spacy
from spacy.tokens import Doc
import skweak
from annotators import full_annotator, labeled_docs_to_pandas
from utilities import make_docs, extract_speeches
import argparse


# CLI parameters
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_path", type=str, default=".\\ParlaMint-HU.txt")
parser.add_argument("-y", "--years", type=str, nargs="*", default=["2013"])

args = parser.parse_args()

# set up spac Doc custom attribute
Doc.set_extension("attrs", default={}, force=True)


# extract the speeches
print("Extracting speeches...")
df_speech = extract_speeches(args.dataset_path, args.years)

# load spacy NLP module; disable all components to speed up processing
print("Loading NLP model...")
nlp = spacy.load(
    "hu_core_news_md",
    disable=[
        "tok2vec",
        "senter",
        "tagger",
        "morphologizer",
        "lookup_lemmatizer",
        "trainable_lemmatizer",
        "parser",
        "ner",
    ],
)


# convert the dataframe to sentence-sized docs
print("Making docs...")
docs = list(make_docs(nlp, df_speech, drop_text_column=False))


# apply the annotator to my docs
print("Annotating docs...")
annotated_docs = list(full_annotator.pipe(docs))

# consolidate it using a model
print("Creating unified HMM model...")
hmm = skweak.generative.HMM("hmm", labels=["PRO", "CONTRA", "NEUTRAL"])
hmm.fit(annotated_docs)

# fit the labels
print("Fitting docs to HMM...")
fitted_docs = list(hmm.pipe(annotated_docs))

# save as a pandas CSV
output_file = f"outputs/labeled-{'-'.join(args.years)}.csv"
print(f"Saving dataset to `{output_file}`...")
df_fitted_docs = labeled_docs_to_pandas(fitted_docs, True)
df_fitted_docs.to_csv(output_file)

print(
    f"Weak supervision done for dataset {args.dataset_path} for years `{', '.join(args.years)}`"
)

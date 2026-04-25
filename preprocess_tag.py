import spacy
from spacy.tokens import Doc
import skweak
from utilities import make_docs, extract_speeches, create_file_name
import argparse
import importlib
import time
from .evaluate_tags import run_lf_diagnostics

# set up spac Doc custom attribute
Doc.set_extension("attrs", default={}, force=True)


def preprocess_tag(
    dataset_path: str = ".\\ParlaMint-HU.txt",
    years: list[str] = ["2014"],
    start_date: str | None = None,
    end_date: str | None = None,
    suffix: str = "",
    *,
    annotators_module="annotators",
    auto_evalute=True,
):
    start = time.time()
    annotators = importlib.import_module(annotators_module)
    full_annotator = annotators.full_annotator
    labeled_docs_to_pandas = annotators.labeled_docs_to_pandas

    # extract the speeches
    print("Extracting speeches...")
    df_speech = extract_speeches(
        dataset_path, years, start_date=start_date, end_date=end_date
    )

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
    file_name = create_file_name("labeled", years, suffix, "csv")
    output_file = f"outputs/{file_name}"

    print(f"Saving dataset to `{output_file}`...")
    df_fitted_docs = labeled_docs_to_pandas(fitted_docs, True)
    df_fitted_docs.to_csv(output_file)

    print(
        f"Weak supervision done for dataset `{dataset_path}`"
        + f" for years `{', '.join(years)}` in {(time.time() - start):.2f} seconds."
    )

    if auto_evalute:
        run_lf_diagnostics(output_file)


if __name__ == "__main__":
    # CLI parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, default=".\\ParlaMint-HU.txt")
    parser.add_argument("-y", "--years", type=str, nargs="*", default=["2013"])
    parser.add_argument("-s", "--start_date", type=str, default=None)
    parser.add_argument("-e", "--end_date", type=str, default=None)
    parser.add_argument("-a", "--annotators_module", type=str, default="annotators")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument(
        "--auto_evaluate", action=argparse.BooleanOptionalAction, default=True
    )

    args = parser.parse_args()

    preprocess_tag(
        args.dataset_path,
        args.years,
        args.start_date,
        args.end_date,
        args.suffix,
        annotators_module=args.annotators_module,
        auto_evalute=args.auto_evaluate,
    )

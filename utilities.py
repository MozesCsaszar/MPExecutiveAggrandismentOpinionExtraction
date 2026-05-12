import os
import pandas as pd
from pathlib import Path
import spacy
from spacy.tokens import Doc, DocBin
from spacy.language import Language
import re
from blingfire import text_to_sentences
from tqdm.auto import tqdm

tqdm.pandas()

# set up spac Doc custom attribute
Doc.set_extension("attrs", default={}, force=True)


def create_file_name(prefix: str, years: list[str], suffix: str, extension: str):
    if len(suffix) > 0:
        suffix = "-" + suffix
    return f"{prefix}-{'-'.join(years)}{suffix}.{extension}"


def extract_speeches(
    root: str,
    years: list[str],
    *,
    filter_chairman: bool = True,
    filter_guest: bool = True,
    convert_dates: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Read in the text data found in the ParliaMint dataset, from path to folder `root`,
    of the form `<PATH_TO_DATASET>\\ParlaMint-<LANG>.txt` and read in all speaches from
    years (represented as subfoders in the dataset) specified in `years`.
    """
    df_speech = pd.DataFrame()
    for year in years:
        path = Path(root) / year
        # loop through all the meta files
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename.endswith("meta-en.tsv"):
                # load using pandas
                df_meta = pd.read_csv(path / file, sep="\t", header=0)
                # load actual text
                df_text = pd.read_csv(
                    path / file.replace("-meta-en.tsv", ".txt"),
                    sep="\t",
                    names=["ID", "Text"],
                )
                # merge based on ID
                df_full = df_meta.merge(df_text, how="inner", on="ID")
                # concatenate with full dataframe
                df_speech = pd.concat([df_speech, df_full])

    # remove irrelevant entries
    # filter out stuff that I won't be processing
    if filter_chairman:
        df_speech = df_speech[(df_speech["Speaker_role"] != "Chairperson")]
    if filter_guest:
        df_speech = df_speech[(df_speech["Speaker_role"] != "Guest")]

    # filter out dates outside of the time range specified
    if start_date is None:
        start_date = df_speech["Date"].min()
    if end_date is None:
        end_date = df_speech["Date"].max()

    df_speech = df_speech[
        (df_speech["Date"] >= start_date) & (df_speech["Date"] <= end_date)
    ]

    # do some data cleaning
    df_speech["Speaker_birth"] = pd.to_numeric(
        df_speech["Speaker_birth"], errors="coerce"
    )
    df_speech["Speaker_minister"] = df_speech["Speaker_minister"].map(
        lambda x: False if x == "notMinister" else True
    )
    df_speech["Speaker_MP"] = df_speech["Speaker_MP"].map(
        lambda x: True if x == "MP" else False
    )
    if convert_dates:
        df_speech["Date"] = pd.to_datetime(df_speech["Date"], format="%Y-%m-%d")

    # remove text annotations (clapping, shouting etc.)
    df_speech["Text"] = df_speech["Text"].apply(
        lambda x: re.sub(r"\[\[[^\[]*\]\]", "", x)
    )

    df_speech["Full_text"] = df_speech["Text"].copy()

    df_speech["Text"] = df_speech["Text"].progress_apply(
        lambda t: text_to_sentences(t).split("\n")
    )
    df_speech = df_speech.explode("Text").reset_index(drop=True)

    # make the ID unique per sentence
    df_speech["ID"] = (
        df_speech["ID"] + "-" + (df_speech.groupby("ID").cumcount() + 1).apply(str)
    )

    return df_speech


# [[ChatGPT]]
def nr_words(doc: Doc) -> int:
    """
    Count the number of words in the SpaCy Doc `doc`.
    """
    return len([token for token in doc if not token.is_punct and not token.is_space])


BATCH_SIZE = 128


def make_docs(
    nlp: Language,
    df: pd.DataFrame,
    *,
    text_column_name="Text",
    drop_text_column: bool = True,
):
    """
    Make sentence-level documents from each text column entry of the dataframe `df`,
    using the `nlp` NLP model provided by spacy.
    """
    texts = df[text_column_name]
    if drop_text_column:
        df = df.drop(columns=[text_column_name])
    metas = df.to_dict(orient="records")

    for doc, meta in zip(
        tqdm(
            nlp.pipe(texts, batch_size=BATCH_SIZE),
            total=len(metas),
        ),
        metas,
    ):

        doc._.attrs = meta
        yield doc


def save_docs(
    docs: list[Doc], years: list[str], filename: str = "nlpd", suffix: str = ""
):
    # save doc nlp component
    doc_bin = DocBin(store_user_data=False)
    for doc in docs:
        doc_bin.add(doc)

    doc_bin.to_disk(create_file_name(f"outputs/{filename}", years, suffix, "docbin"))

    # save metadata
    df = pd.DataFrame([doc._.attrs for doc in docs])
    df.to_csv(create_file_name(f"outputs/{filename}", years, suffix, "csv"))


def load_docs(
    nlp: Language, years: list[str], filename: str = "nlpd", suffix: str = ""
) -> list[Doc] | None:
    try:
        # extract metadata
        df = pd.read_csv(create_file_name(f"outputs/{filename}", years, suffix, "csv"))
        metas = df.to_dict(orient="records")

        # recove docs
        doc_bin = DocBin()
        doc_bin = doc_bin.from_disk(
            create_file_name(f"outputs/{filename}", years, suffix, "docbin")
        )
        docs = list(doc_bin.get_docs(nlp.vocab))

        # annotate with metadata
        for meta, doc in tqdm(zip(metas, docs)):
            doc._.attrs = meta

        return docs
    except FileNotFoundError:
        return None

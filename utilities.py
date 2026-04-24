import os
import pandas as pd
from pathlib import Path
import spacy
from spacy.tokens import Doc
import re
from blingfire import text_to_sentences
from tqdm.auto import tqdm

tqdm.pandas()


def extract_speeches(
    root: str,
    years: list[str],
    *,
    filter_chairman: bool = True,
    filter_guest: bool = True,
    convert_dates: bool = False,
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

    df_speech["Text"] = df_speech["Text"].progress_apply(
        lambda t: text_to_sentences(t).split("\n")
    )
    df_speech = df_speech.explode("Text").reset_index(drop=True)

    return df_speech


# [[ChatGPT]]
def nr_words(doc: Doc) -> int:
    """
    Count the number of words in the SpaCy Doc `doc`.
    """
    return len([token for token in doc if not token.is_punct and not token.is_space])


BATCH_SIZE = 128


def make_docs(
    nlp: spacy.language.Language,
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

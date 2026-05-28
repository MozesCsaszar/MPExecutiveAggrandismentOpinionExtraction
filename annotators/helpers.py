from spacy.tokens import Doc
import skweak
import pandas as pd
import os
from pathlib import Path
import re


# extract the labels for spans
def extract_label(doc: Doc, layer: str) -> str | None:
    labels = [x.label_ for x in skweak.utils.get_spans(doc, [layer])]
    return labels[0] if len(labels) == 1 else None


def default_metadata_columns() -> list[str]:
    return ["Text", "ID"]


def labeled_docs_to_pandas(
    fitted_docs: list[Doc], full_metadata: bool = False
) -> pd.DataFrame:
    # convert to dict representation; first add keys for metadata fields
    attrs_keys = list(fitted_docs[0]._.attrs.keys())
    docs_dict = {key: [] for key in attrs_keys}

    # add keys for LF result lists
    span_keys = list(fitted_docs[0].spans.keys())
    docs_dict = {**{key: [] for key in span_keys}, **docs_dict}

    # convert to this format
    for doc in fitted_docs:
        for key, value in doc._.attrs.items():
            docs_dict[key].append(value)
        for key in span_keys:
            docs_dict[key].append(extract_label(doc, key))

    # drop irrelevant metadata columns
    if not full_metadata:
        keep_metadata = default_metadata_columns()
        for key in attrs_keys:
            if key not in keep_metadata:
                docs_dict.pop(key)

    # convert to pandas dataframe and return
    return pd.DataFrame(docs_dict)


def load_label_files(
    years: list[str],
    suffix: str,
    prefix: str = "_llm-labeled",
    path: str = "llm_labeled",
):
    df = pd.DataFrame()

    # create the folder and filename regex
    folder = Path(path)
    if suffix != "":
        suffix = "_" + suffix
    filename_regex = rf"{prefix}-{'-'.join(years)}.*{suffix}\.csv"
    print("File name regex:", filename_regex)

    # loop through all the meta files
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        # if the filename matches the regex, load it
        if re.match(filename_regex, filename):
            # load the dataframe part using pandas
            df_part = pd.read_csv(folder / file, header=0, index_col=0)
            # concatenate with full dataframe
            if len(df) != 0:
                df = pd.concat([df, df_part])
            else:
                df = df_part

    return df

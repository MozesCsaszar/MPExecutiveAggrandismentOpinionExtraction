from spacy.tokens import Doc
import pandas as pd
import re
from pathlib import Path
import os
import skweak


def load_llm_labels(
    years: list[str],
    suffix: str,
    prefix: str = "_llm-labeled",
    path: str = "llm_labeled",
) -> dict:
    df = pd.DataFrame()

    # create the folder and filename regex
    folder = Path(path)
    filename_regex = rf"{prefix}-{'-'.join(years)}-\d*-\d*_{suffix}\.csv"

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

    if len(df) == 0:
        return {}
    # keep only the columns that are needed
    df = df[["label", "ID"]]
    # convert to ID:label format and return
    return {d["ID"]: d["label"] for d in df.to_dict(orient="records")}


def create_llm_annotator(
    name: str,
    years: list[str],
    suffix: str,
    prefix: str = "_llm-labeled",
    path: str = "llm_labeled",
):
    label_dict = load_llm_labels(years, suffix, prefix, path)

    def lf(doc: Doc):
        label = label_dict.get(doc._.attrs["ID"])
        if label:
            yield doc[0].i, doc[-1].i + 1, label

    lf.__name__ = name
    return skweak.heuristics.FunctionAnnotator(name, lf)  # type: ignore


lf_gemini_flash_v1_annotator = create_llm_annotator(
    "lf_gemini_flash_v1_annotator", ["2017"], "gemini_flash"
)


llm_annotators = skweak.base.CombinedAnnotator()
llm_annotators.add_annotator(lf_gemini_flash_v1_annotator)

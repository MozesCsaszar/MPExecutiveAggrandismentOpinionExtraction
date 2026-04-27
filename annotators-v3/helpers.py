from spacy.tokens import Doc, Span
import skweak
import pandas as pd


# extract the labels for spans
def extract_label(doc: Doc, layer: str) -> str | None:
    labels = [x.label_ for x in skweak.utils.get_spans(doc, [layer])]
    return labels[0] if len(labels) == 1 else None


def _lf_preprocess(doc: Doc):
    return doc.text.lower()


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

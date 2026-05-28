from spacy.tokens import Doc
import skweak
from .helpers import load_label_files


def load_file_based_labels(
    years: list[str],
    suffix: str,
    prefix: str = "_llm-labeled",
    path: str = "llm_labeled",
) -> dict:
    df = load_label_files(years, suffix, prefix, path)

    if len(df) == 0:
        return {}
    # keep only the columns that are needed
    df = df[["label", "ID"]]
    # convert to ID:label format and return
    return {d["ID"]: d["label"] for d in df.to_dict(orient="records")}


def create_file_based_annotator(
    name: str,
    years: list[str],
    suffix: str,
    prefix: str = "_llm-labeled",
    path: str = "llm_labeled",
):
    label_dict = load_file_based_labels(years, suffix, prefix, path)
    print("Annotator name:", name, "Number of entries:", len(label_dict))

    def lf(doc: Doc):
        label = label_dict.get(doc._.attrs["ID"])
        if label:
            yield doc[0].i, doc[-1].i + 1, label

    lf.__name__ = name
    return skweak.heuristics.FunctionAnnotator(name, lf)  # type: ignore


lf_gemini_flash_v1_annotator = create_file_based_annotator(
    "lf_gemini_flash_v1_annotator", ["2017"], "gemini_flash"
)

similarity_annotators = []
for label in ["pro", "contra", "neutral"]:
    annotator = create_file_based_annotator(
        f"lf_discovery_{label}",
        ["2017"],
        "",
        prefix=f"discovery_{label}",
        path="outputs",
    )

    similarity_annotators.append(annotator)

file_annotators = skweak.base.CombinedAnnotator()
file_annotators.add_annotator(lf_gemini_flash_v1_annotator)
file_annotators.add_annotators(*similarity_annotators)

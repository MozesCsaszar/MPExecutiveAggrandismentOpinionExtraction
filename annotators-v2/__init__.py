from annotators.pro import pro_annotator
from annotators.contra import contra_annotator
from annotators.neutral import neutral_annotator
from annotators.helpers import labeled_docs_to_pandas
import skweak

full_annotator = skweak.base.CombinedAnnotator()
full_annotator.add_annotators(pro_annotator, contra_annotator, neutral_annotator)

LABELS = ["PRO", "CON", "NEUTRAL"]


__all__ = [
    "pro_annotator",
    "contra_annotator",
    "neutral_annotator",
    "full_annotator",
    "labeled_docs_to_pandas",
    "LABELS",
]

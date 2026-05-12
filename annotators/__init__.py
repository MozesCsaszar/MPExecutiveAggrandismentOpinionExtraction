from .pro import pro_annotator
from .contra import contra_annotator
from .neutral import neutral_annotator
from .helpers import labeled_docs_to_pandas
from .file_based import file_annotators
import skweak

full_annotator = skweak.base.CombinedAnnotator()
full_annotator.add_annotators(
    pro_annotator, contra_annotator, neutral_annotator, file_annotators
)

LABELS = ["PRO", "CON", "NEUTRAL"]


__all__ = [
    "pro_annotator",
    "contra_annotator",
    "neutral_annotator",
    "full_annotator",
    "labeled_docs_to_pandas",
    "LABELS",
]

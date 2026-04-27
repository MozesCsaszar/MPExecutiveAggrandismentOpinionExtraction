from .pro import pro_annotator
from .contra import contra_annotator
from .neutral import neutral_annotator
from .helpers import labeled_docs_to_pandas
from .llm import llm_annotators
import skweak

full_annotator = skweak.base.CombinedAnnotator()
full_annotator.add_annotators(
    pro_annotator, contra_annotator, neutral_annotator, llm_annotators
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

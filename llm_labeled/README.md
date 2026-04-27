# General
The following sections describe the labeling method used to obtain LLM classified PRO, NEUTRAL and CONTRA sentence level predictions.
## _llm-labeled-2017-*
- speeches from 2017 labeled using different LLMs with different prompts
- they are labeled on the sentence level, while giving the LLM the speech the sentence is part of as context
### _llm-labeled-2017-<range_start>-<range_end>_gemini_flash.csv
- the speeches were filtered so that they are longer than 50, but shorter than 2000 characters to prevent using too many credits on this first run
- the sentences were also filtered to remove too short ones (less than 30 characters) for the same reason
- these files were labeled by gemini flash, the default option provided by Kaggle's benchmarks functionality
- labeling 1000 entries costs roughly \$4-\$5 with this model in credits
- the following prompt was used:
```python
prompt = f"""
You are an expert political analyst specializing in Hungarian politics. Your task is to classify a specific sentence from a parliamentary speech
regarding its stance on 'executive aggrandisement'. Executive aggrandisement refers to the process by which
an executive branch expands its power or undermines democratic institutions.

Based on the full speech provided, determine if the specified sentence expresses a stance that is:
- "PRO": The Member of Parliament (MP) supports or advocates for executive aggrandisement.
- "CONTRA": The MP is against or criticizes executive aggrandisement.
- "NEUTRAL": The sentence does not clearly support or oppose executive aggrandisement, or it discusses it factually without taking a side.

You must provide your assessment as a JSON object with three fields: 'label', 'confidence', and 'reason'.

Full Parliamentary Speech:
---
{full_speech}
---

Sentence to Classify:
---
{sentence}
---

Your 'reason' should be concise (no more than 25 words), in English and explicitly reference how the full speech and the specific sentence led to your conclusion.
"""
```
- find the notebook [here](https://www.kaggle.com/code/mozescsaszar/executive-aggrandisement-opinion-labeling/edit)
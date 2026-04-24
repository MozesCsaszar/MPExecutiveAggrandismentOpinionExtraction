# set up labeling stuff
label2id = {label: i for i, label in enumerate(["CONTRA", "NEUTRAL", "PRO"])}
id2label = {label2id[label]: label for label in label2id}

# model stuff
model_id = "FacebookAI/xlm-roberta-base"
seed = 42

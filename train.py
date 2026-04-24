from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import numpy as np
import evaluate
import random
import argparse
import pandas as pd
from consts import model_id, label2id, id2label


def create_dataset(
    df: pd.DataFrame,
    pro_frac=0.3,
    con_frac=0.3,
    neut_frac=0.4,
    *,
    seed: int = 42,
    label_name: str = "hmm",
    test_size: float = 0.1,
):
    if pro_frac + con_frac + neut_frac > 1:
        raise ValueError("Error: fraction sum should be 1!")

    print("Calculating label distribution...")
    data = (
        df[["Text", label_name]]
        .rename(columns={"Text": "text", label_name: "label"})
        .to_dict("records")
    )
    contra_data = [d for d in data if d["label"] == "CONTRA"]
    neutral_data = [d for d in data if d["label"] == "NEUTRAL"]
    pro_data = [d for d in data if d["label"] == "PRO"]

    nr_con = len(contra_data)
    nr_neut = len(neutral_data)
    nr_pro = len(pro_data)

    random.seed(seed)

    print(f"TOTAL CONTRA: {nr_con}, NEUTRAL: {nr_neut}, PRO: {nr_pro}")
    # if there are less pro and con than neutral, balance that way
    if nr_pro * pro_frac + nr_con * con_frac < nr_neut * neut_frac:
        nr_neut = int((nr_pro + nr_con) / (pro_frac + con_frac) * neut_frac)

        neutral_data = random.sample(neutral_data, nr_neut)
    # otherwise, balance it the other way around
    else:
        nr_pro = int(nr_neut / neut_frac * pro_frac)
        nr_con = int(nr_neut / neut_frac * con_frac)

        pro_data = random.sample(pro_data, nr_pro)
        contra_data = random.sample(contra_data, nr_con)

    print(f"SELECTED CONTRA: {nr_con}, NEUTRAL: {nr_neut}, PRO: {nr_pro}")
    # full data
    full_data = [*pro_data, *contra_data, *neutral_data]

    # TODO: Add real validation (or remove it completely)
    return Dataset.from_list(full_data).train_test_split(test_size=test_size, seed=seed)


def train(
    dataset_path: str,
    *,
    label_name: str = "hmm",
    seed: int = 42,
):
    # create the dataset
    df_fitted_docs = pd.read_csv(dataset_path)
    dataset = create_dataset(df_fitted_docs, label_name=label_name, seed=seed)

    # save the training dataset
    # TODO: Bebetter naming
    dataset.save_to_disk("./outputs/train")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # preprocess the data to get it into the common format
    # TODO: Refactor this to be in-line with the classifier one
    def preprocess_function(speeches):
        tokenized = tokenizer(speeches["text"], truncation=True)
        tokenized["labels"] = [label2id[label] for label in speeches["label"]]
        return tokenized

    print("Tokenizing speeches...")
    tokenized_speeches = dataset.map(preprocess_function, batched=True)
    tokenized_speeches = tokenized_speeches.select_columns(
        ["input_ids", "attention_mask", "labels"]
    )

    # create the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # define validation metrics
    print("Setting up evaluation...")
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision", zero_division=0)
    recall = evaluate.load("recall", zero_division=0)
    f1_metric = evaluate.load("f1", zero_division=0)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels),
            "precision": precision.compute(
                predictions=predictions, references=labels, average=None  # type: ignore
            )["precision"].tolist(),
            "recall": recall.compute(
                predictions=predictions, references=labels, average=None  # type: ignore
            )["recall"].tolist(),
            "f1": f1_metric.compute(
                predictions=predictions, references=labels, average=None  # type: ignore
            )["f1"].tolist(),
        }

    # load the model
    print("Loading the model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    # set up the training
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir="xlm-r-ea",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_speeches["train"],
        eval_dataset=tokenized_speeches["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # train the model
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    # CLI parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-l", "--label_name", type=str, default="hmm")

    args = parser.parse_args()
    train(args.dataset_path, label_name=args.label_name, seed=args.seed)

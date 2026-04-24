from transformers import (
    pipeline,
)
from datasets import Dataset
from tqdm.auto import tqdm
import argparse
from utilities import extract_speeches


def tag_new_data(
    dataset_path: str = ".\\ParlaMint-HU.txt",
    years: list[str] = ["2017"],
    model_path: str = "xlm-r-ea/checkpoint-6",
    batch_size: int = 256,
):
    # load the model
    print("Loading the model...")
    classifier = pipeline(
        "text-classification",
        model=model_path,
        dtype="auto",
        batch_size=batch_size,
    )

    # load the pandas df to be classified
    print("Extracting speeches...")
    df_speech = extract_speeches(dataset_path, years)
    dataset = Dataset.from_pandas(df_speech)

    # classify the new dataset
    print("Classifying the data...")
    texts = dataset["Text"]
    labels, scores = [], []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        preds = classifier(batch, batch_size=batch_size, truncation=True)
        labels.extend([p["label"] for p in preds])
        scores.extend([p["score"] for p in preds])

    dataset = dataset.add_column("label", labels)
    dataset = dataset.add_column("score", scores)

    # convert to pandas
    print("Saving the speeches...")
    df_speech = dataset.to_pandas()

    # index by Speaker_ID and timestamp
    df_speech = df_speech.set_index(["Speaker_ID", "Date", "ID"]).sort_index()

    # save the data
    df_speech.to_csv("./outputs/hu_2017.csv")
    print("Done!")


if __name__ == "__main__":
    # CLI parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, default=".\\ParlaMint-HU.txt")
    parser.add_argument("-y", "--years", type=str, nargs="*", default=["2017"])
    parser.add_argument("-m", "--model_path", type=str, default="xlm-r-ea/checkpoint-6")
    parser.add_argument("-b", "--batch_size", type=int, default=256)

    args = parser.parse_args()

    tag_new_data(args.dataset_path, args.years, args.model_path, args.batch_size)

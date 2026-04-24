import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", type=str)
parser.add_argument("-e", "--max_examples", type=int, default=5)
parser.add_argument("-x", "--extra_lf_cols", type=str, nargs="+", default=["hmm"])

args = parser.parse_args()


# [[ChatGPT]] + modified by me
def run_lf_diagnostics(
    file_path: str = "outputs/preprocessed.csv",
    max_examples: int = 5,
    extra_lf_cols=["hmm"],
):
    print("Reading data...")
    # the full dataframe
    df_full = pd.read_csv(file_path)
    lf_cols = [col for col in df_full.columns if col.find("lf") != -1]

    # the dataframe only containing LFs
    all_cols = [*lf_cols, *extra_lf_cols]
    df = df_full[lf_cols]

    # --- Coverage ---
    print("Calculating coverage...")
    coverage = (df_full[lf_cols].notnull().any(axis=1)).mean()

    # --- Per-LF stats ---
    print("Calculating per-LF stats...")
    lf_stats = []
    for lf_col in tqdm(all_cols, total=len(all_cols), postfix="LFs"):
        col = df_full[lf_col]
        fired = col.notnull().sum()
        coverage_lf = fired / len(df_full)

        label_counts = col.value_counts().to_dict()

        lf_stats.append(
            {
                "LF": lf_col,
                "Coverage": coverage_lf,
                "Fires": fired,
                "PRO": label_counts.get("PRO", 0),
                "CONTRA": label_counts.get("CONTRA", 0),
                "NEUTRAL": label_counts.get("NEUTRAL", 0),
                "Abstain": len(df_full) - fired,
            }
        )

    lf_stats_df = pd.DataFrame(lf_stats).sort_values(by="Coverage", ascending=False)

    # Overlap
    print("Calculating overlap...")
    num_lfs_fired = df.notnull().sum(axis=1)
    overlap_mean = num_lfs_fired.mean()

    # Conflict
    print("Calculating conflict...")

    def has_conflict(row):
        labels = set(row.dropna())
        return "PRO" in labels and "CONTRA" in labels

    conflict_rate = df.apply(has_conflict, axis=1).mean()

    # Label distribution (raw LF votes)
    all_votes = df.stack()
    label_dist = all_votes.value_counts(normalize=True)

    # Extract Examples
    print("Extracting examples...")
    examples = {
        "high_conflict": [],
        "pro_examples": [],
        "con_examples": [],
        "neutral_examples": [],
    }

    # TODO: Speed this up by direct query
    for i, row in df.iterrows():
        labels = list(row.dropna())
        text = df_full.iloc[[i]]["Text"]

        if "PRO" in labels and "CONTRA" in labels:
            if len(examples["high_conflict"]) < max_examples:
                examples["high_conflict"].append(text)

        elif "PRO" in labels:
            if len(examples["pro_examples"]) < max_examples:
                examples["pro_examples"].append(text)

        elif "CONTRA" in labels:
            if len(examples["con_examples"]) < max_examples:
                examples["con_examples"].append(text)

        elif "NEUTRAL" in labels:
            if len(examples["neutral_examples"]) < max_examples:
                examples["neutral_examples"].append(text)

    # Output statistics
    print("\n" + "=" * 60)
    print("GLOBAL METRICS")
    print("=" * 60)
    print(f"Coverage (≥1 LF fires): {coverage:.3f}")
    print(f"Average #LFs per doc: {overlap_mean:.2f}")
    print(f"Conflict rate (PRO vs CONTRA): {conflict_rate:.3f}")

    print("\nLabel distribution (LF votes):")
    print(label_dist.to_string())

    print("\n" + "=" * 60)
    print("LF STATISTICS")
    print("=" * 60)
    print(lf_stats_df.to_string(index=False))

    # TODO: Re-enable this if ever the need arises
    # print("\n" + "=" * 60)
    # print("LF CORRELATIONS")
    # print("=" * 60)
    # mapping = {"CONTRA": -1, "NEUTRAL": 0, "PRO": 1}
    # df_encoded = df.replace(mapping)
    # corr = df_encoded.corr()
    # print(corr)

    print("\n" + "=" * 60)

    print("EXAMPLES")
    print("=" * 60)

    for k, v in examples.items():
        if len(v) > 0:
            print(f"\n{k.upper()}:")
            for ex in v:
                text = ex.values[0]
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"- {text}")
        else:
            print(f"\nNO {k.upper()}...")

    return df, lf_stats_df


print("Running diagnostics...")
run_lf_diagnostics(args.dataset_path, args.max_examples, args.extra_lf_cols)

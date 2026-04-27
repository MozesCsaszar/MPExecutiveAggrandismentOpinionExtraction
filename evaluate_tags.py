import pandas as pd
import argparse
from tqdm.auto import tqdm


def extract_lf_stats(df: pd.DataFrame, all_label_cols: list[str]):
    print("Calculating per-LF stats...")
    lf_stats = []
    for lf_col in tqdm(all_label_cols, total=len(all_label_cols), postfix="LFs"):
        col = df[lf_col]
        fired = col.notnull().sum()
        coverage_lf = fired / len(df)

        label_counts = col.value_counts().to_dict()

        lf_stats.append(
            {
                "LF": lf_col,
                "Coverage": coverage_lf,
                "Fires": fired,
                "PRO": label_counts.get("PRO", 0),
                "CONTRA": label_counts.get("CONTRA", 0),
                "NEUTRAL": label_counts.get("NEUTRAL", 0),
                "Abstain": len(df) - fired,
            }
        )

    return pd.DataFrame(lf_stats).sort_values(by="Coverage", ascending=False)


def _format_examples(df: pd.DataFrame, max_examples: int = 5):
    return (
        df.head(max_examples)
        .apply(lambda r: f"[[{r['ID']}]] {r['Text']}", axis=1)
        .tolist()
    )


# [[ChatGPT]]+
def extract_examples(
    df: pd.DataFrame,
    max_examples: int = 5,
    final_label_col: str = "hmm",
):
    df_high_conflict = df[
        (df.isin(["PRO"]).any(axis=1)) & (df.isin(["CONTRA"]).any(axis=1))
    ]
    df_pro = df[df[final_label_col] == "PRO"]
    df_contra = df[df[final_label_col] == "CONTRA"]
    df_neutral = df[df[final_label_col] == "NEUTRAL"]

    return {
        "high_conflict": _format_examples(df_high_conflict, max_examples),
        "pro_examples": _format_examples(df_pro, max_examples),
        "con_examples": _format_examples(df_contra, max_examples),
        "neutral_examples": _format_examples(df_neutral, max_examples),
    }


# [[ChatGPT]] + modified by me
def run_lf_diagnostics(
    file_path: str = "outputs/preprocessed.csv",
    max_examples: int = 5,
    extra_lf_cols=["hmm"],
):
    print("Reading data...")
    # the full dataframe
    df_full = pd.read_csv(file_path, low_memory=False)
    lf_cols = [col for col in df_full.columns if col.find("lf") != -1]

    # the dataframe only containing LFs
    all_label_cols = [*lf_cols, *extra_lf_cols]
    df = df_full[lf_cols]

    # --- Coverage ---
    print("Calculating coverage...")
    coverage = (df_full[lf_cols].notnull().any(axis=1)).mean()

    # --- Per-LF stats ---
    lf_stats_df = extract_lf_stats(df_full, all_label_cols)

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
    examples = extract_examples(df_full, max_examples)

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
            for text in v:
                print(f" - {text}")
        else:
            print(f"\nNO {k.upper()}...")

    return df, lf_stats_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("-e", "--max_examples", type=int, default=5)
    parser.add_argument("-x", "--extra_lf_cols", type=str, nargs="+", default=["hmm"])

    args = parser.parse_args()

    run_lf_diagnostics(args.dataset_path, args.max_examples, args.extra_lf_cols)

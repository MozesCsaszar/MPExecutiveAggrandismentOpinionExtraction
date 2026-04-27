from utilities import extract_speeches, create_file_name
import argparse
from annotators.consts import PRO_POWER, CON_DEMO, CON_RULE_OF_LAW, INSTITUTIONS


KEYWORDS = [*PRO_POWER, *CON_DEMO, *CON_RULE_OF_LAW, *INSTITUTIONS]


def extract_for_llm(
    root: str = ".\\ParlaMint-HU.txt",
    years: list[str] = ["2017"],
    nr_lines: int = 100,
    starting_line: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
):
    print("Extracting speeches...")
    df = extract_speeches(root, years, start_date=start_date, end_date=end_date)

    # filter by keyword appearance
    print(f"Filtering {len(df)} speeches based on keywords...")
    pattern = "|".join(KEYWORDS)
    df = df[df["Full_text"].str.contains(pattern, case=False, na=False)]
    # remove speeches that are way too long or short
    print(f"Filtering {len(df)} speeches based on full text dimensions...")
    df = df[df["Full_text"].str.len().between(100, 2000, inclusive="both")]
    print(f"Filtering {len(df)} speeches based on sentence dimensions...")
    df = df[df["Text"].str.len().between(30, 2000, inclusive="both")]

    # filter down to first <nr_lines>
    print(f"Transforming {len(df)} speeches to {nr_lines}...")
    df = df.iloc[starting_line : starting_line + nr_lines]

    # drop irrelevant columns
    df = df[["Full_text", "Text_ID", "ID", "Text"]]

    # rename columns to look better
    df = df.rename(columns={"Full_text": "full_speech", "Text": "sentence"})
    return df


if __name__ == "__main__":
    # CLI parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, default=".\\ParlaMint-HU.txt")
    parser.add_argument("-y", "--years", type=str, nargs="*", default=["2017"])
    parser.add_argument("-s", "--start_date", type=str, default=None)
    parser.add_argument("-e", "--end_date", type=str, default=None)
    parser.add_argument("-n", "--nr_lines", type=int, default=100)
    parser.add_argument("-f", "--starting_line", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument(
        "--auto_evaluate", action=argparse.BooleanOptionalAction, default=True
    )

    args = parser.parse_args()

    df = extract_for_llm(
        args.dataset_path,
        args.years,
        args.nr_lines,
        args.starting_line,
        args.start_date,
        args.end_date,
    )

    output_path = create_file_name(
        ".\\outputs\\llm",
        args.years,
        f"{args.starting_line}-{args.starting_line + args.nr_lines}",
        "csv",
    )
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)

    print("Done.")

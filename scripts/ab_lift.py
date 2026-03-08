"""Compare channel lift from campaign performance data."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/raw/shipment_delays.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if "channel" in df.columns and "conversions" in df.columns:
        lift = df.groupby("channel")["conversions"].mean().sort_values(ascending=False)
    else:
        lift = df.select_dtypes("number").mean().sort_values(ascending=False)
    print("Channel lift (mean conversions / metric):")
    print(lift.to_string())


if __name__ == "__main__":
    main()

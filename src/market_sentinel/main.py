from __future__ import annotations

import argparse

from tabulate import tabulate

from .analysis import run_sector_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Market Sentinel")
    parser.add_argument("--strong", type=int, default=3, help="Number of strongest sectors")
    parser.add_argument("--weak", type=int, default=2, help="Number of weakest sectors")
    parser.add_argument("--picks", type=int, default=8, help="Number of stock suggestions")
    args = parser.parse_args()

    strong, weak, picks = run_sector_report(top_strong=args.strong, top_weak=args.weak)
    picks = picks.head(args.picks)

    print("\\nStrongest sectors (YTD):")
    print(tabulate(strong, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    print("\\nWeakest sectors (YTD):")
    print(tabulate(weak, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    print("\\nSuggested stocks from strongest sectors:")
    print(tabulate(picks, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))


if __name__ == "__main__":
    main()

"""Regenerate all publication tables from packaged verified results."""

from publication_utils import prepare_tables


if __name__ == "__main__":
    metrics, comparisons, statistics = prepare_tables()
    print(f"Generated {6} publication CSV tables.")
    print(f"Metrics rows: {len(metrics)}; comparison rows: {len(comparisons)}; statistical rows: {len(statistics)}.")

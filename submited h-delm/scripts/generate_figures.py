"""Regenerate publication figures from packaged predictions and histories."""

from publication_utils import generate_figures


if __name__ == "__main__":
    count = generate_figures()
    print(f"Generated {count} figures in PNG, vector PDF, and SVG formats.")

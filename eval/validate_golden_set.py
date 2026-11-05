"""CLI and validation utility for Golden QA evaluation dataset.

Validates:
1. Pydantic schema conformance (schema, unique IDs, non-empty questions and answers).
2. Answerable vs unanswerable consistency and presence of gold_urls.
3. Existence of gold_urls in the indexed PostHog documentation corpus.
4. Target distribution requirements (for final dataset validation):
   - Total items >= 40
   - Single-chunk questions ~60% (50% - 70%)
   - Multi-chunk questions >= 8
   - Unanswerable questions >= 8
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

from eval.schema import GoldenDataset
from src.config import BM25_INDEX_PATH


def normalize_url(url: str) -> str:
    """Normalize a documentation URL by stripping whitespace, query parameters, fragment, and trailing slash."""
    clean = url.strip().split("#")[0].split("?")[0].strip().rstrip("/")
    return clean


def load_corpus_urls(bm25_index_path: Optional[Path] = None) -> set[str]:
    """Extract all distinct canonical URLs from the indexed BM25 / chunk lookup corpus."""
    index_file = bm25_index_path or BM25_INDEX_PATH
    if not index_file.exists():
        raise FileNotFoundError(f"Corpus index not found at: {index_file}")

    with open(index_file, "rb") as f:
        _, chunk_lookup = pickle.load(f)

    corpus_urls = {normalize_url(chunk.url) for chunk in chunk_lookup if getattr(chunk, "url", None)}
    return corpus_urls


def validate_golden_file(
    file_path: str | Path,
    check_distribution: bool = False,
    corpus_urls: Optional[set[str]] = None,
    skip_corpus_check: bool = False,
) -> tuple[bool, list[str]]:
    """Programmatically validate a golden dataset file.

    Args:
        file_path: Path to dataset JSON file.
        check_distribution: Whether to strictly enforce distribution criteria.
        corpus_urls: Optional preloaded set of valid corpus URLs.
        skip_corpus_check: If True, bypass checking gold_urls against indexed corpus.

    Returns:
        tuple (is_valid, error_messages)
    """
    errors: list[str] = []
    p = Path(file_path)

    if not p.exists():
        return False, [f"File not found: {p}"]

    try:
        dataset = GoldenDataset.from_file(p)
    except Exception as exc:
        return False, [f"Schema / parsing validation failed: {exc}"]

    # Validate corpus URLs
    if not skip_corpus_check:
        if corpus_urls is None:
            try:
                corpus_urls = load_corpus_urls()
            except Exception as exc:
                errors.append(f"Failed to load corpus index for URL verification: {exc}")
                corpus_urls = None

        if corpus_urls is not None:
            norm_corpus = {normalize_url(u) for u in corpus_urls}
            for item in dataset.items:
                if item.answerable:
                    for url in item.gold_urls:
                        norm = normalize_url(url)
                        if norm not in norm_corpus:
                            errors.append(
                                f"Item '{item.id}': URL '{url}' not found in indexed corpus"
                            )

    # Validate distribution targets if requested
    if check_distribution:
        dist_res = dataset.validate_distribution()
        if not dist_res.valid:
            errors.extend(dist_res.errors)

    is_valid = len(errors) == 0
    return is_valid, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Golden QA dataset schema, URLs, and distribution."
    )
    parser.add_argument("path", help="Path to golden dataset JSON file to validate")
    parser.add_argument(
        "--check-distribution",
        action="store_true",
        help="Enforce final distribution criteria (>=40 items, ~60% single, >=8 multi, >=8 unanswerable)",
    )
    parser.add_argument(
        "--skip-corpus-check",
        action="store_true",
        help="Skip verification of gold_urls against the indexed corpus",
    )
    parser.add_argument(
        "--corpus-index",
        type=Path,
        default=None,
        help="Custom path to bm25_index.pkl for URL verification",
    )

    args = parser.parse_args()

    dataset_path = Path(args.path)
    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found at: {dataset_path}")
        return 1

    print(f"Validating dataset: {dataset_path}")

    # Load corpus URLs if needed
    corpus_urls = None
    if not args.skip_corpus_check:
        try:
            print("Loading indexed corpus URLs...")
            corpus_urls = load_corpus_urls(args.corpus_index)
            print(f"Loaded {len(corpus_urls)} distinct canonical documentation URLs from corpus.")
        except Exception as exc:
            print(f"WARNING: Could not load corpus index ({exc}). Proceeding without corpus check.")

    is_valid, errors = validate_golden_file(
        file_path=dataset_path,
        check_distribution=args.check_distribution,
        corpus_urls=corpus_urls,
        skip_corpus_check=args.skip_corpus_check or (corpus_urls is None),
    )

    try:
        dataset = GoldenDataset.from_file(dataset_path)
        counts = dataset.type_counts()
        total = len(dataset)
        single = counts["single-chunk"]
        multi = counts["multi-chunk"]
        unanswerable = counts["unanswerable"]
        single_pct = (single / total * 100) if total > 0 else 0.0

        print(f"\n--- Dataset Summary ---")
        print(f"Total items:         {total}")
        print(f"Single-chunk items:  {single} ({single_pct:.1f}%)")
        print(f"Multi-chunk items:   {multi}")
        print(f"Unanswerable items:  {unanswerable}")
    except Exception:
        pass

    if not is_valid:
        print(f"\n❌ Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\n✅ Dataset validation PASSED successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

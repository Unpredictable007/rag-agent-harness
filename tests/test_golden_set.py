"""Unit tests for Golden QA dataset schema, validation, and CLI checks."""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from eval.schema import GoldenDataset, GoldenItem
from eval.validate_golden_set import normalize_url, validate_golden_file


def test_valid_single_chunk_item():
    item = GoldenItem(
        id="gold-001",
        question="How do I enable session recording in PostHog?",
        reference_answer="Enable it under Project Settings > Session Replay and configure sampling.",
        gold_urls=["https://posthog.com/docs/session-replay"],
        type="single-chunk",
        answerable=True,
        notes="Grounded in session-replay overview",
    )
    assert item.id == "gold-001"
    assert item.answerable is True
    assert item.type == "single-chunk"
    assert len(item.gold_urls) == 1


def test_valid_multi_chunk_item():
    item = GoldenItem(
        id="gold-002",
        question="How do feature flags interact with experiments?",
        reference_answer="Experiments use multivariate feature flags under the hood to assign variants.",
        gold_urls=[
            "https://posthog.com/docs/feature-flags",
            "https://posthog.com/docs/experiments",
        ],
        type="multi-chunk",
        answerable=True,
    )
    assert item.type == "multi-chunk"
    assert len(item.gold_urls) == 2


def test_valid_unanswerable_item():
    item = GoldenItem(
        id="gold-003",
        question="What is the internal FPGA clock speed requirement for PostHog?",
        reference_answer="PostHog is software analytics and does not require or document FPGA hardware clock rates.",
        gold_urls=[],
        type="unanswerable",
        answerable=False,
    )
    assert item.answerable is False
    assert item.type == "unanswerable"
    assert len(item.gold_urls) == 0


def test_invalid_empty_fields():
    with pytest.raises(ValidationError):
        GoldenItem(
            id="  ",
            question="Question?",
            reference_answer="Answer",
            gold_urls=["https://posthog.com/docs/test"],
            type="single-chunk",
            answerable=True,
        )

    with pytest.raises(ValidationError):
        GoldenItem(
            id="gold-004",
            question="",
            reference_answer="Answer",
            gold_urls=["https://posthog.com/docs/test"],
            type="single-chunk",
            answerable=True,
        )

    with pytest.raises(ValidationError):
        GoldenItem(
            id="gold-005",
            question="Question?",
            reference_answer="",
            gold_urls=["https://posthog.com/docs/test"],
            type="single-chunk",
            answerable=True,
        )


def test_invalid_answerable_consistency():
    # answerable=True but type is unanswerable
    with pytest.raises(ValidationError, match="cannot have type 'unanswerable'"):
        GoldenItem(
            id="gold-006",
            question="Question?",
            reference_answer="Answer",
            gold_urls=["https://posthog.com/docs/test"],
            type="unanswerable",
            answerable=True,
        )

    # answerable=True but gold_urls is empty
    with pytest.raises(ValidationError, match="must contain at least one gold_url"):
        GoldenItem(
            id="gold-007",
            question="Question?",
            reference_answer="Answer",
            gold_urls=[],
            type="single-chunk",
            answerable=True,
        )

    # answerable=True with whitespace URL
    with pytest.raises(ValidationError, match="cannot contain empty URL strings"):
        GoldenItem(
            id="gold-008",
            question="Question?",
            reference_answer="Answer",
            gold_urls=["   "],
            type="single-chunk",
            answerable=True,
        )

    # answerable=False but type is single-chunk
    with pytest.raises(ValidationError, match="must have type 'unanswerable'"):
        GoldenItem(
            id="gold-009",
            question="Question?",
            reference_answer="Explanation of unanswerable",
            gold_urls=[],
            type="single-chunk",
            answerable=False,
        )


def test_dataset_unique_ids():
    item1 = GoldenItem(
        id="gold-001",
        question="Q1",
        reference_answer="A1",
        gold_urls=["https://posthog.com/docs/q1"],
        type="single-chunk",
        answerable=True,
    )
    item2 = GoldenItem(
        id="gold-001",
        question="Q2",
        reference_answer="A2",
        gold_urls=["https://posthog.com/docs/q2"],
        type="single-chunk",
        answerable=True,
    )
    with pytest.raises(ValidationError, match="Duplicate item IDs found"):
        GoldenDataset(items=[item1, item2])


def test_dataset_get_and_type_counts():
    item1 = GoldenItem(
        id="gold-001",
        question="Q1",
        reference_answer="A1",
        gold_urls=["https://posthog.com/docs/q1"],
        type="single-chunk",
        answerable=True,
    )
    item2 = GoldenItem(
        id="gold-002",
        question="Q2",
        reference_answer="A2",
        gold_urls=["https://posthog.com/docs/q2"],
        type="multi-chunk",
        answerable=True,
    )
    item3 = GoldenItem(
        id="gold-003",
        question="Q3",
        reference_answer="Unanswerable",
        gold_urls=[],
        type="unanswerable",
        answerable=False,
    )
    ds = GoldenDataset(items=[item1, item2, item3])

    assert len(ds) == 3
    assert ds.get_item("gold-002") == item2
    assert ds.get_item("gold-999") is None

    counts = ds.type_counts()
    assert counts == {"single-chunk": 1, "multi-chunk": 1, "unanswerable": 1}


def test_dataset_serialization_roundtrip(tmp_path: Path):
    item1 = GoldenItem(
        id="gold-001",
        question="Q1",
        reference_answer="A1",
        gold_urls=["https://posthog.com/docs/q1"],
        type="single-chunk",
        answerable=True,
    )
    ds = GoldenDataset(items=[item1])
    file_path = tmp_path / "test_dataset.json"

    ds.to_file(file_path)
    loaded = GoldenDataset.from_file(file_path)
    assert len(loaded) == 1
    assert loaded[0].id == "gold-001"

    # Also test loading from a bare list format
    bare_list_path = tmp_path / "bare_list.json"
    with open(bare_list_path, "w", encoding="utf-8") as f:
        json.dump([item1.model_dump()], f)
    loaded_bare = GoldenDataset.from_file(bare_list_path)
    assert len(loaded_bare) == 1
    assert loaded_bare[0].id == "gold-001"


def test_dataset_distribution_validation():
    # Create 40 items: 24 single (60%), 8 multi (20%), 8 unanswerable (20%)
    items = []
    idx = 1
    for _ in range(24):
        items.append(
            GoldenItem(
                id=f"gold-{idx:03d}",
                question=f"Single question {idx}",
                reference_answer=f"Answer {idx}",
                gold_urls=[f"https://posthog.com/docs/doc-{idx}"],
                type="single-chunk",
                answerable=True,
            )
        )
        idx += 1

    for _ in range(8):
        items.append(
            GoldenItem(
                id=f"gold-{idx:03d}",
                question=f"Multi question {idx}",
                reference_answer=f"Answer {idx}",
                gold_urls=[
                    f"https://posthog.com/docs/doc-{idx}a",
                    f"https://posthog.com/docs/doc-{idx}b",
                ],
                type="multi-chunk",
                answerable=True,
            )
        )
        idx += 1

    for _ in range(8):
        items.append(
            GoldenItem(
                id=f"gold-{idx:03d}",
                question=f"Unanswerable question {idx}",
                reference_answer=f"Explanation {idx}",
                gold_urls=[],
                type="unanswerable",
                answerable=False,
            )
        )
        idx += 1

    ds = GoldenDataset(items=items)
    res = ds.validate_distribution()
    assert res.valid is True
    assert res.total == 40
    assert res.single_chunk_count == 24
    assert res.multi_chunk_count == 8
    assert res.unanswerable_count == 8
    assert len(res.errors) == 0

    # Test failure: only 10 items
    small_ds = GoldenDataset(items=items[:10])
    res_small = small_ds.validate_distribution()
    assert res_small.valid is False
    assert any("below target minimum (40)" in err for err in res_small.errors)


def test_normalize_url():
    assert normalize_url("https://posthog.com/docs/feature-flags#rollout") == "https://posthog.com/docs/feature-flags"
    assert normalize_url("https://posthog.com/docs/session-replay/  ") == "https://posthog.com/docs/session-replay"
    assert normalize_url("https://posthog.com/docs/events") == "https://posthog.com/docs/events"


def test_validate_golden_file_with_corpus_urls(tmp_path: Path):
    corpus_urls = {
        "https://posthog.com/docs/feature-flags",
        "https://posthog.com/docs/experiments",
    }

    valid_item = GoldenItem(
        id="gold-001",
        question="How do flags work?",
        reference_answer="Grounded answer.",
        gold_urls=["https://posthog.com/docs/feature-flags#intro"],
        type="single-chunk",
        answerable=True,
    )
    dataset = GoldenDataset(items=[valid_item])
    file_path = tmp_path / "valid.json"
    dataset.to_file(file_path)

    # Valid with matching corpus URL (ignoring fragment)
    valid, errors = validate_golden_file(
        file_path=file_path,
        check_distribution=False,
        corpus_urls=corpus_urls,
    )
    assert valid is True
    assert len(errors) == 0

    # Invalid URL not in corpus
    invalid_item = GoldenItem(
        id="gold-002",
        question="Unknown doc question?",
        reference_answer="Answer.",
        gold_urls=["https://posthog.com/docs/non-existent-article"],
        type="single-chunk",
        answerable=True,
    )
    invalid_dataset = GoldenDataset(items=[invalid_item])
    invalid_file = tmp_path / "invalid.json"
    invalid_dataset.to_file(invalid_file)

    valid, errors = validate_golden_file(
        file_path=invalid_file,
        check_distribution=False,
        corpus_urls=corpus_urls,
    )
    assert valid is False
    assert any("not found in indexed corpus" in err for err in errors)


def test_generated_candidates_file():
    candidates_path = Path(__file__).resolve().parent.parent / "eval" / "candidates.json"
    if not candidates_path.exists():
        pytest.skip("eval/candidates.json not generated yet")

    dataset = GoldenDataset.from_file(candidates_path)
    assert len(dataset) == 60

    counts = dataset.type_counts()
    assert counts["single-chunk"] == 36
    assert counts["multi-chunk"] == 12
    assert counts["unanswerable"] == 12

    dist = dataset.validate_distribution()
    assert dist.valid is True
    assert len(dist.errors) == 0


def test_golden_set_integrity_and_distribution():
    import hashlib

    golden_path = Path(__file__).resolve().parent.parent / "eval" / "golden_set.json"
    hash_path = Path(__file__).resolve().parent.parent / "eval" / "golden_set.hash"

    assert golden_path.exists(), "eval/golden_set.json missing"
    assert hash_path.exists(), "eval/golden_set.hash missing"

    # 1. Cryptographic SHA-256 verification
    sha256 = hashlib.sha256(golden_path.read_bytes()).hexdigest()
    expected_hash = hash_path.read_text(encoding="utf-8").strip()
    assert sha256 == expected_hash, f"Hash mismatch! Expected {expected_hash}, got {sha256}"

    # 2. Schema and distribution targets verification
    dataset = GoldenDataset.from_file(golden_path)
    assert len(dataset) == 60
    dist = dataset.validate_distribution()
    assert dist.valid is True
    assert len(dist.errors) == 0

    # 3. Stratified split verification
    dev_items = dataset.filter_by_split("dev")
    holdout_items = dataset.filter_by_split("holdout")
    assert len(dev_items) == 45
    assert len(holdout_items) == 15
    assert len(dev_items) + len(holdout_items) == 60

    # 4. Dev answerable check: exactly 36 items for retrieval benchmarks
    dev_answerable = [it for it in dev_items if it.answerable]
    assert len(dev_answerable) == 36


def test_unanswerable_disallows_gold_urls():
    with pytest.raises(ValidationError, match="cannot contain gold_urls"):
        GoldenItem(
            id="gold-099",
            question="Can PostHog bake cookies?",
            reference_answer="No, PostHog is analytics software.",
            gold_urls=["https://posthog.com/docs/something"],
            type="unanswerable",
            answerable=False,
        )


def test_normalize_url_with_query_params_and_spaces():
    assert normalize_url("https://posthog.com/docs/libraries/python?tab=async") == "https://posthog.com/docs/libraries/python"
    assert normalize_url("https://posthog.com/docs/events #anchor") == "https://posthog.com/docs/events"
    assert normalize_url(" https://posthog.com/docs/session-replay/  ") == "https://posthog.com/docs/session-replay"



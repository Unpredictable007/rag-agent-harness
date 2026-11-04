"""Pydantic schemas and dataset container for Golden QA evaluation dataset."""

import json
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


QuestionType = Literal["single-chunk", "multi-chunk", "unanswerable"]
DatasetSplit = Literal["dev", "holdout"]


class GoldenItem(BaseModel):
    """A single QA evaluation item grounded in PostHog documentation or marked unanswerable."""

    id: str = Field(description="Unique identifier, e.g. gold-001 or cand-001")
    question: str = Field(description="User question")
    reference_answer: str = Field(
        description="Grounded reference answer, or explicit explanation for unanswerable questions"
    )
    gold_urls: list[str] = Field(
        default_factory=list,
        description="Ground truth source documentation URLs grounding the reference answer",
    )
    type: QuestionType = Field(
        description="Evaluation question type: single-chunk, multi-chunk, or unanswerable"
    )
    answerable: bool = Field(
        description="Whether the question can be answered from the documentation corpus"
    )
    split: Optional[DatasetSplit] = Field(
        default=None,
        description="Evaluation split: 'dev' for tuning and iterative evaluation, 'holdout' for frozen test",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional notes, synthesis rationale, chunk references, or candidate annotations",
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Item id cannot be empty or whitespace")
        return s

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Question cannot be empty or whitespace")
        return s

    @field_validator("reference_answer")
    @classmethod
    def validate_reference_answer(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Reference answer cannot be empty or whitespace")
        return s

    @model_validator(mode="after")
    def validate_answerable_consistency(self) -> "GoldenItem":
        if self.answerable:
            if self.type == "unanswerable":
                raise ValueError("Item with answerable=True cannot have type 'unanswerable'")
            if not self.gold_urls:
                raise ValueError("Item with answerable=True must contain at least one gold_url")
            for url in self.gold_urls:
                if not url or not url.strip():
                    raise ValueError("gold_urls cannot contain empty URL strings")
        else:
            if self.type != "unanswerable":
                raise ValueError(
                    f"Item with answerable=False must have type 'unanswerable', got '{self.type}'"
                )
            if self.gold_urls:
                raise ValueError("Item with answerable=False cannot contain gold_urls")
        return self


class DistributionValidationResult(BaseModel):
    """Result of dataset distribution validation checks."""

    valid: bool
    total: int
    single_chunk_count: int
    single_chunk_ratio: float
    multi_chunk_count: int
    unanswerable_count: int
    errors: list[str] = Field(default_factory=list)


class GoldenDataset(BaseModel):
    """Container model for a collection of Golden QA items with validation helpers."""

    items: list[GoldenItem] = Field(default_factory=list, description="List of Golden QA items")

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "GoldenDataset":
        seen: set[str] = set()
        duplicates: list[str] = []
        for item in self.items:
            if item.id in seen:
                duplicates.append(item.id)
            seen.add(item.id)
        if duplicates:
            unique_dups = sorted(set(duplicates))
            raise ValueError(f"Duplicate item IDs found in dataset: {unique_dups}")
        return self

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[GoldenItem]:
        return iter(self.items)

    def __getitem__(self, index: int) -> GoldenItem:
        return self.items[index]

    def get_item(self, item_id: str) -> Optional[GoldenItem]:
        """Lookup an item by its unique ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def type_counts(self) -> dict[str, int]:
        """Return counts of items grouped by question type."""
        counts = {"single-chunk": 0, "multi-chunk": 0, "unanswerable": 0}
        for item in self.items:
            counts[item.type] = counts.get(item.type, 0) + 1
        return counts

    def filter_by_split(self, split: DatasetSplit) -> list[GoldenItem]:
        """Return items belonging to a specific evaluation split ('dev' or 'holdout')."""
        return [it for it in self.items if it.split == split]

    def validate_distribution(
        self,
        min_total: int = 40,
        min_single_ratio: float = 0.50,
        max_single_ratio: float = 0.70,
        min_multi: int = 8,
        min_unanswerable: int = 8,
    ) -> DistributionValidationResult:
        """Validate whether dataset meets target distribution criteria for final golden evaluation."""
        total = len(self.items)
        counts = self.type_counts()
        single = counts["single-chunk"]
        multi = counts["multi-chunk"]
        unanswerable = counts["unanswerable"]
        single_ratio = (single / total) if total > 0 else 0.0

        errors: list[str] = []
        if total < min_total:
            errors.append(f"Total items ({total}) below target minimum ({min_total})")
        if total > 0 and not (min_single_ratio <= single_ratio <= max_single_ratio):
            errors.append(
                f"Single-chunk ratio ({single_ratio:.1%}) outside target ~60% range "
                f"({min_single_ratio:.0%} - {max_single_ratio:.0%})"
            )
        if multi < min_multi:
            errors.append(f"Multi-chunk count ({multi}) below target minimum ({min_multi})")
        if unanswerable < min_unanswerable:
            errors.append(f"Unanswerable count ({unanswerable}) below target minimum ({min_unanswerable})")

        return DistributionValidationResult(
            valid=len(errors) == 0,
            total=total,
            single_chunk_count=single,
            single_chunk_ratio=single_ratio,
            multi_chunk_count=multi,
            unanswerable_count=unanswerable,
            errors=errors,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "GoldenDataset":
        """Load dataset from a JSON file (accepts either a dict with 'items' or a bare list)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Golden dataset file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return cls(items=[GoldenItem(**it) for it in data])
        elif isinstance(data, dict):
            if "items" in data:
                return cls(**data)
            raise ValueError(f"Dataset JSON in {p} must contain an 'items' key")
        raise ValueError(f"Unexpected JSON format in {p}: expected list or dict with 'items'")

    def to_file(self, path: str | Path, indent: int = 2) -> None:
        """Write dataset to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=indent, ensure_ascii=False)
            f.write("\n")

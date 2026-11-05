"""Generate Golden QA candidate evaluation dataset by sampling documentation chunks.

Produces:
1. eval/candidates.json: GoldenDataset containing ~60 candidates (36 single-chunk, 12 multi-chunk, 12 unanswerable).
2. eval/candidates_for_review.md: Formatted markdown document for human review and curation.
"""

import argparse
import json
import logging
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

from eval.schema import GoldenDataset, GoldenItem
from src.config import BM25_INDEX_PATH
from src.ingest import DocumentChunk
from src.llm import generate_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_FILE = REPO_ROOT / "eval" / ".candidates_cache.json"


def load_cached_items() -> dict[str, GoldenItem]:
    """Load previously cached candidate items if available."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cached = {it["id"]: GoldenItem(**it) for it in data}
        logger.info("Loaded %d previously cached items from %s", len(cached), CACHE_FILE)
        return cached
    except Exception as exc:
        logger.warning("Could not read cache file %s: %s", CACHE_FILE, exc)
        return {}


def save_cached_item(item: GoldenItem) -> None:
    """Append or update a single item in the local candidate cache file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cached = load_cached_items()
    cached[item.id] = item
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump([it.model_dump() for it in cached.values()], f, indent=2, ensure_ascii=False)


def parse_json_from_llm(raw_text: str) -> Any:
    """Robustly parse JSON object or array from LLM response text."""
    # Check for markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if fence_match:
        target_str = fence_match.group(1).strip()
    else:
        # Find outermost brackets
        first_brace = raw_text.find("{")
        first_bracket = raw_text.find("[")
        if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
            last_bracket = raw_text.rfind("]")
            target_str = raw_text[first_bracket : last_bracket + 1]
        elif first_brace != -1:
            last_brace = raw_text.rfind("}")
            target_str = raw_text[first_brace : last_brace + 1]
        else:
            target_str = raw_text.strip()

    return json.loads(target_str)


def load_all_chunks(bm25_path: Optional[Path] = None) -> list[DocumentChunk]:
    """Load all indexed document chunks from the BM25 index file."""
    path = bm25_path or BM25_INDEX_PATH
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found at: {path}")
    logger.info("Loading chunk lookup from %s...", path)
    with open(path, "rb") as f:
        _, chunk_lookup = pickle.load(f)
    chunks = list(chunk_lookup)
    logger.info("Loaded %d total chunks.", len(chunks))
    return chunks


def sample_single_chunks(
    chunks: list[DocumentChunk],
    count: int = 36,
    seed: int = 42,
) -> list[DocumentChunk]:
    """Sample substantive chunks across diverse documentation domains for single-chunk questions."""
    rng = random.Random(seed)

    # Substantive chunk filter: 130 to 450 tokens, non-empty text, excluding API references / changelogs
    filtered = [
        c
        for c in chunks
        if 130 <= c.token_count <= 450
        and not c.url.endswith("/api")
        and not any(x in c.title.lower() for x in ["release notes", "changelog", "migration guide", "deprecated"])
        and len(c.text.strip()) > 250
    ]

    # Category quota distribution for 36 items
    category_quotas = [
        ("product-analytics", 4),
        ("feature-flags", 4),
        ("experiments", 3),
        ("session-replay", 4),
        ("web-analytics", 3),
        ("data-pipelines", 4),
        ("data-warehouse", 3),
        ("error-tracking", 3),
        ("libraries", 4),
        ("data", 4),
    ]

    selected: list[DocumentChunk] = []
    used_urls: set[str] = set()

    for category, quota in category_quotas:
        pool = [
            c
            for c in filtered
            if c.doc_path.startswith(category + "/") and c.url not in used_urls
        ]
        rng.shuffle(pool)
        taken = 0
        for c in pool:
            if c.url not in used_urls:
                selected.append(c)
                used_urls.add(c.url)
                taken += 1
                if taken >= quota:
                    break
        logger.info("Sampled %d chunks for category '%s' (quota: %d)", taken, category, quota)

    # If short of 36 due to deduplication, pick from remaining substantive chunks across other areas
    if len(selected) < count:
        remaining = [c for c in filtered if c.url not in used_urls]
        rng.shuffle(remaining)
        needed = count - len(selected)
        for c in remaining[:needed]:
            selected.append(c)
            used_urls.add(c.url)

    selected = selected[:count]
    logger.info("Selected %d single-chunk candidates across %d distinct URLs.", len(selected), len(used_urls))
    return selected


def sample_multi_chunk_pairs(
    chunks: list[DocumentChunk],
    count: int = 12,
    exclude_chunk_ids: Optional[set[str]] = None,
    seed: int = 42,
) -> list[tuple[DocumentChunk, DocumentChunk]]:
    """Sample 12 pairs of related chunks from complementary docs or sections in the same domain."""
    rng = random.Random(seed)
    exclude = exclude_chunk_ids or set()

    filtered = [
        c
        for c in chunks
        if 110 <= c.token_count <= 420
        and c.chunk_id not in exclude
        and not c.url.endswith("/api")
        and not any(x in c.title.lower() for x in ["release notes", "changelog", "migration guide"])
        and len(c.text.strip()) > 200
    ]

    domains = [
        "feature-flags",
        "experiments",
        "session-replay",
        "product-analytics",
        "web-analytics",
        "surveys",
        "error-tracking",
        "data-warehouse",
        "data-pipelines",
        "libraries",
        "data",
        "ai-observability",
    ]

    pairs: list[tuple[DocumentChunk, DocumentChunk]] = []
    used_chunk_ids: set[str] = set(exclude)

    for domain in domains:
        domain_chunks = [
            c
            for c in filtered
            if c.doc_path.startswith(domain + "/") and c.chunk_id not in used_chunk_ids
        ]
        # Group by document URL
        by_doc = defaultdict(list)
        for c in domain_chunks:
            by_doc[c.url].append(c)

        doc_urls = list(by_doc.keys())
        rng.shuffle(doc_urls)

        pair_found = False
        # Try to find two chunks from two different related documents
        if len(doc_urls) >= 2:
            c1 = by_doc[doc_urls[0]][0]
            c2 = by_doc[doc_urls[1]][0]
            pairs.append((c1, c2))
            used_chunk_ids.add(c1.chunk_id)
            used_chunk_ids.add(c2.chunk_id)
            pair_found = True
        elif len(doc_urls) == 1 and len(by_doc[doc_urls[0]]) >= 2:
            # Fall back to two distinct sections of the same comprehensive document
            c1 = by_doc[doc_urls[0]][0]
            c2 = by_doc[doc_urls[0]][1]
            pairs.append((c1, c2))
            used_chunk_ids.add(c1.chunk_id)
            used_chunk_ids.add(c2.chunk_id)
            pair_found = True

        if pair_found and len(pairs) >= count:
            break

    logger.info("Selected %d multi-chunk pairs across domains.", len(pairs))
    return pairs[:count]


def generate_single_chunk_candidates(
    chunks: list[DocumentChunk],
    start_idx: int = 1,
    batch_size: int = 3,
) -> list[GoldenItem]:
    """Generate single-chunk Q&A candidate items using the LLM in batches."""
    logger.info("Generating %d single-chunk candidates in batches of %d...", len(chunks), batch_size)
    cached_all = load_cached_items()
    items: list[GoldenItem] = []

    system_prompt = (
        "You are an expert technical documentation evaluator creating a rigorous Golden QA benchmark "
        "for PostHog documentation search. Generate realistic, specific questions that a developer or "
        "product manager using PostHog would ask. Ground reference answers STRICTLY in the provided text."
    )

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        # Check if all items in this batch are already cached
        batch_ids = [f"cand-{start_idx + i + offset:03d}" for offset in range(len(batch))]
        if all(bid in cached_all for bid in batch_ids):
            logger.info("Reusing cached single-chunk batch %d/%d (items %s)...", batch_num, total_batches, batch_ids)
            for bid in batch_ids:
                items.append(cached_all[bid])
            continue

        logger.info(
            "Processing single-chunk batch %d/%d (items %d to %d)...",
            batch_num,
            total_batches,
            i + 1,
            i + len(batch),
        )

        chunk_prompts = []
        for idx_in_batch, c in enumerate(batch, start=1):
            chunk_prompts.append(
                f"--- CHUNK {idx_in_batch} ---\n"
                f"URL: {c.url}\n"
                f"Title: {c.title}\n"
                f"Heading: {c.heading_path}\n"
                f"Content:\n{c.text}\n"
            )

        prompt = (
            "Generate a realistic developer/user question and a factual reference answer for EACH of the "
            "following documentation chunks.\n"
            "Requirements:\n"
            "1. Each question must sound natural and realistic (do not copy phrasing verbatim from titles).\n"
            "2. Each question must be strictly answerable using ONLY its corresponding chunk content.\n"
            "3. Reference answer must directly answer the question using only facts stated in that chunk.\n\n"
            + "\n".join(chunk_prompts)
            + f"\nReturn ONLY a JSON array with exactly {len(batch)} objects. Each object must have:\n"
            "- \"chunk_index\": integer (1, 2, ...)\n"
            "- \"question\": string\n"
            "- \"reference_answer\": string"
        )

        parsed = None
        for attempt in range(2):
            try:
                response = generate_text(prompt=prompt, system_prompt=system_prompt, max_tokens=8192)
                parsed = parse_json_from_llm(response.text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    break
            except Exception as exc:
                logger.warning("Attempt %d failed for single-chunk batch %d: %s", attempt + 1, batch_num, exc)
                time.sleep(2)

        if not isinstance(parsed, list):
            raise ValueError(f"Failed to parse valid JSON array from LLM for batch {batch_num}")

        for list_idx, obj in enumerate(parsed):
            c_idx = int(obj.get("chunk_index", obj.get("index", list_idx + 1))) - 1
            if 0 <= c_idx < len(batch):
                c = batch[c_idx]
                item_id = f"cand-{start_idx + len(items):03d}"
                item = GoldenItem(
                    id=item_id,
                    question=str(obj["question"]).strip(),
                    reference_answer=str(obj["reference_answer"]).strip(),
                    gold_urls=[c.url],
                    type="single-chunk",
                    answerable=True,
                    notes=f"Source: {c.chunk_id} | Title: {c.title} | Headings: {c.heading_path}",
                )
                items.append(item)
                save_cached_item(item)

    logger.info("Generated/loaded %d single-chunk candidates successfully.", len(items))
    return items


def generate_multi_chunk_candidates(
    pairs: list[tuple[DocumentChunk, DocumentChunk]],
    start_idx: int = 37,
) -> list[GoldenItem]:
    """Generate multi-chunk Q&A candidate items using the LLM (1 pair per prompt for maximum focus)."""
    logger.info("Generating %d multi-chunk candidates (1 pair per prompt)...", len(pairs))
    cached_all = load_cached_items()
    items: list[GoldenItem] = []

    system_prompt = (
        "You are an expert technical evaluator creating multi-hop synthesis evaluation questions "
        "for PostHog documentation search. Each question must require synthesizing information "
        "from BOTH provided chunks. Reference answers must integrate facts from both chunks."
    )

    for idx, (c1, c2) in enumerate(pairs, start=1):
        item_id = f"cand-{start_idx + len(items):03d}"
        if item_id in cached_all:
            logger.info("Reusing cached multi-chunk item %s...", item_id)
            items.append(cached_all[item_id])
            continue

        logger.info("Processing multi-chunk pair %d/%d (%s)...", idx, len(pairs), item_id)

        prompt = (
            f"Given these two related documentation chunks:\n\n"
            f"--- CHUNK 1 ---\n"
            f"URL: {c1.url}\n"
            f"Title: {c1.title} ({c1.heading_path})\n"
            f"Content:\n{c1.text}\n\n"
            f"--- CHUNK 2 ---\n"
            f"URL: {c2.url}\n"
            f"Title: {c2.title} ({c2.heading_path})\n"
            f"Content:\n{c2.text}\n\n"
            "TASK:\n"
            "Generate a realistic developer/user synthesis question and reference answer.\n"
            "Requirements:\n"
            "1. The question must require information from BOTH Chunk 1 and Chunk 2 to be completely answered.\n"
            "2. Neither Chunk 1 alone nor Chunk 2 alone should be sufficient to answer it.\n"
            "3. The reference answer must synthesize the relevant facts from both chunks accurately.\n\n"
            "Return ONLY a JSON object with keys:\n"
            "\"question\": string\n"
            "\"reference_answer\": string\n"
            "\"synthesis_notes\": string explaining why both chunks are needed"
        )

        parsed = None
        for attempt in range(2):
            try:
                response = generate_text(prompt=prompt, system_prompt=system_prompt, max_tokens=8192)
                raw_parsed = parse_json_from_llm(response.text)
                if isinstance(raw_parsed, list) and len(raw_parsed) > 0:
                    parsed = raw_parsed[0]
                elif isinstance(raw_parsed, dict):
                    parsed = raw_parsed
                if parsed and "question" in parsed and "reference_answer" in parsed:
                    break
            except Exception as exc:
                logger.warning("Attempt %d failed for multi-chunk pair %d: %s", attempt + 1, idx, exc)
                time.sleep(2)

        if not parsed:
            raise ValueError(f"Failed to parse valid JSON for multi-chunk pair {idx}")

        urls = [c1.url]
        if c2.url not in urls:
            urls.append(c2.url)

        notes = (
            f"Synthesized from:\n"
            f"  - Chunk 1: {c1.chunk_id} ({c1.url})\n"
            f"  - Chunk 2: {c2.chunk_id} ({c2.url})\n"
            f"Rationale: {parsed.get('synthesis_notes', 'Requires facts from both documentation chunks')}"
        )

        item = GoldenItem(
            id=item_id,
            question=str(parsed["question"]).strip(),
            reference_answer=str(parsed["reference_answer"]).strip(),
            gold_urls=urls,
            type="multi-chunk",
            answerable=True,
            notes=notes,
        )
        items.append(item)
        save_cached_item(item)

    logger.info("Generated/loaded %d multi-chunk candidates successfully.", len(items))
    return items


def generate_unanswerable_candidates(
    start_idx: int = 49,
    count: int = 12,
) -> list[GoldenItem]:
    """Generate unanswerable candidate questions (plausible queries not supported by PostHog docs)."""
    logger.info("Generating %d unanswerable candidates...", count)
    cached_all = load_cached_items()
    items: list[GoldenItem] = []

    # Check if all unanswerable items are already cached
    unans_ids = [f"cand-{start_idx + offset:03d}" for offset in range(count)]
    if all(uid in cached_all for uid in unans_ids):
        logger.info("Reusing cached unanswerable items %s...", unans_ids)
        for uid in unans_ids:
            items.append(cached_all[uid])
        return items

    system_prompt = (
        "You are an expert technical evaluator creating negative test cases for a PostHog documentation search benchmark. "
        "PostHog is an all-in-one product analytics platform (event analytics, web analytics, session replay, feature flags, "
        "A/B experiments, surveys, error tracking, and data warehouse).\n"
        "Generate realistic user questions asking about capabilities that PostHog does NOT support, does NOT do, "
        "and which are NOT covered in PostHog documentation. The questions must sound plausible for enterprise or software users."
    )

    batches_spec = [
        # Batch 1: Hardware, legacy systems, enterprise ERP/billing, audio/biometrics
        (
            "Generate 6 distinct, plausible questions asking about capabilities that PostHog does NOT support:\n"
            "1. Automotive CAN bus or IoT hardware sensor direct signal ingestion protocol.\n"
            "2. Automated payroll tax calculation or HR shift deductions based on active user hours.\n"
            "3. Biometric audio pitch analysis or facial emotion recognition in session replay.\n"
            "4. Native SAP ECC / SAP GUI screen recording driver for desktop session recording.\n"
            "5. Legacy IBM mainframe COBOL SDK for analytics tracking.\n"
            "6. Native Jira sprint velocity estimation and automated story point poker.\n\n"
            "Return ONLY a JSON array with 6 objects, each with:\n"
            "- \"question\": realistic question\n"
            "- \"reference_explanation\": clear, professional explanation of why PostHog does not support or document this\n"
            "- \"rationale\": brief note on why this is an effective negative benchmark case"
        ),
        # Batch 2: Cryptography, blockchain, CAD/design, video streaming, native database internals
        (
            "Generate 6 distinct, plausible questions asking about capabilities that PostHog does NOT support:\n"
            "1. Quantum random number generator key distribution in self-hosted PostHog.\n"
            "2. Automatic triggering of blockchain smart contract execution on user funnel dropoff.\n"
            "3. Native Apple iOS Swift UIKit storyboard visual code generator compiled from analytics funnels.\n"
            "4. In-database Apache Cassandra distributed commit coordinator engine replacing ClickHouse.\n"
            "5. Direct Salesforce CPQ complex pricing rule and quote approval workflow engine.\n"
            "6. Real-time video conferencing audio/video codec transcoding in the ingestion pipeline.\n\n"
            "Return ONLY a JSON array with 6 objects, each with:\n"
            "- \"question\": realistic question\n"
            "- \"reference_explanation\": clear, professional explanation of why PostHog does not support or document this\n"
            "- \"rationale\": brief note on why this is an effective negative benchmark case"
        ),
    ]

    for batch_idx, prompt in enumerate(batches_spec, start=1):
        target_ids = [f"cand-{start_idx + len(items) + offset:03d}" for offset in range(6)]
        if all(tid in cached_all for tid in target_ids):
            for tid in target_ids:
                items.append(cached_all[tid])
            continue

        logger.info("Processing unanswerable batch %d/2...", batch_idx)
        parsed = None
        for attempt in range(2):
            try:
                response = generate_text(prompt=prompt, system_prompt=system_prompt, max_tokens=8192)
                parsed = parse_json_from_llm(response.text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    break
            except Exception as exc:
                logger.warning("Attempt %d failed for unanswerable batch %d: %s", attempt + 1, batch_idx, exc)
                time.sleep(2)

        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array for unanswerable batch {batch_idx}")

        for obj in parsed:
            item_id = f"cand-{start_idx + len(items):03d}"
            explanation = obj.get("reference_explanation") or obj.get("reference_answer", "")
            item = GoldenItem(
                id=item_id,
                question=str(obj["question"]).strip(),
                reference_answer=str(explanation).strip(),
                gold_urls=[],
                type="unanswerable",
                answerable=False,
                notes=f"Negative evaluation case: {obj.get('rationale', obj.get('notes', 'Unsupported feature'))}",
            )
            items.append(item)
            save_cached_item(item)

    logger.info("Generated/loaded %d unanswerable candidates successfully.", len(items))
    return items[:count]


def format_candidates_markdown(dataset: GoldenDataset) -> str:
    """Format candidate dataset into a clean markdown document for human review."""
    counts = dataset.type_counts()
    lines = [
        "# Candidate Golden Dataset for Human Review",
        "",
        f"**Total Candidates**: {len(dataset)}",
        f"- Single-chunk questions: {counts['single-chunk']}",
        f"- Multi-chunk questions: {counts['multi-chunk']}",
        f"- Unanswerable questions: {counts['unanswerable']}",
        "",
        "## Review Instructions for Evaluator (Artur)",
        "- [ ] Verify that each question sounds realistic for a software engineer or product manager using PostHog.",
        "- [ ] Verify that single-chunk questions are 100% grounded in their source URL.",
        "- [ ] Verify that multi-chunk questions genuinely require synthesis across both listed source URLs.",
        "- [ ] Verify that unanswerable questions represent plausible queries that cannot be answered from PostHog documentation.",
        "- [ ] Check the box `[x]` next to each approved question or edit the reference answer if needed.",
        "",
        "---",
        "",
    ]

    # Section 1: Single-chunk items
    lines.append("## 1. Single-Chunk Questions (36 items)\n")
    for item in dataset.items:
        if item.type == "single-chunk":
            url_str = item.gold_urls[0] if item.gold_urls else "N/A"
            lines.extend(
                [
                    f"### [{item.id}] {item.question[:80]}...",
                    f"- **Type**: `single-chunk` | **Answerable**: `Yes`",
                    f"- **Question**: {item.question}",
                    f"- **Reference Answer**: {item.reference_answer}",
                    f"- **Ground Truth URL**: [{url_str}]({url_str})",
                    f"- **Notes**: {item.notes or 'N/A'}",
                    f"- **Human Verification**:",
                    f"  - [ ] Approved",
                    "",
                ]
            )

    # Section 2: Multi-chunk items
    lines.append("## 2. Multi-Chunk Questions (12 items)\n")
    for item in dataset.items:
        if item.type == "multi-chunk":
            url_lines = "\n".join([f"  - [{u}]({u})" for u in item.gold_urls])
            lines.extend(
                [
                    f"### [{item.id}] {item.question[:80]}...",
                    f"- **Type**: `multi-chunk` | **Answerable**: `Yes`",
                    f"- **Question**: {item.question}",
                    f"- **Reference Answer**: {item.reference_answer}",
                    f"- **Ground Truth URLs**:\n{url_lines}",
                    f"- **Synthesis Notes**:\n```\n{item.notes or 'N/A'}\n```",
                    f"- **Human Verification**:",
                    f"  - [ ] Approved",
                    "",
                ]
            )

    # Section 3: Unanswerable items
    lines.append("## 3. Unanswerable Questions (12 items)\n")
    for item in dataset.items:
        if item.type == "unanswerable":
            lines.extend(
                [
                    f"### [{item.id}] {item.question[:80]}...",
                    f"- **Type**: `unanswerable` | **Answerable**: `No`",
                    f"- **Question**: {item.question}",
                    f"- **Reference Explanation**: {item.reference_answer}",
                    f"- **Ground Truth URLs**: *(None - unanswerable from PostHog docs corpus)*",
                    f"- **Negative Case Rationale**: {item.notes or 'N/A'}",
                    f"- **Human Verification**:",
                    f"  - [ ] Approved",
                    "",
                ]
            )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate candidate golden QA dataset items from PostHog documentation chunks."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "eval" / "candidates.json",
        help="Path to output candidates JSON file",
    )
    parser.add_argument(
        "--output-review",
        type=Path,
        default=REPO_ROOT / "eval" / "candidates_for_review.md",
        help="Path to output markdown review document",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic chunk sampling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample chunks and print plan without invoking LLM",
    )

    args = parser.parse_args()

    chunks = load_all_chunks()
    single_chunks = sample_single_chunks(chunks, count=36, seed=args.seed)
    exclude_ids = {c.chunk_id for c in single_chunks}
    multi_pairs = sample_multi_chunk_pairs(chunks, count=12, exclude_chunk_ids=exclude_ids, seed=args.seed)

    print("\n--- Candidate Generation Plan ---")
    print(f"Single-chunk samples: {len(single_chunks)}")
    print(f"Multi-chunk pairs:    {len(multi_pairs)}")
    print(f"Unanswerable target:  12")
    print(f"Total candidates:     {len(single_chunks) + len(multi_pairs) + 12}")
    print(f"Output JSON:          {args.output_json}")
    print(f"Output Review:        {args.output_review}\n")

    if args.dry_run:
        print("Dry run requested. Exiting without generating candidates.")
        return 0

    single_items = generate_single_chunk_candidates(single_chunks, start_idx=1)
    multi_items = generate_multi_chunk_candidates(multi_pairs, start_idx=len(single_items) + 1)
    unanswerable_items = generate_unanswerable_candidates(
        start_idx=len(single_items) + len(multi_items) + 1, count=12
    )

    all_items = single_items + multi_items + unanswerable_items
    dataset = GoldenDataset(items=all_items)

    logger.info("Writing candidates JSON to %s...", args.output_json)
    dataset.to_file(args.output_json)

    logger.info("Writing candidates review markdown to %s...", args.output_review)
    review_md = format_candidates_markdown(dataset)
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_review, "w", encoding="utf-8") as f:
        f.write(review_md)

    print("\n✅ Candidate generation completed successfully!")
    print(f"Total generated candidates: {len(dataset)}")
    print(f"  - Single-chunk: {len(single_items)}")
    print(f"  - Multi-chunk:  {len(multi_items)}")
    print(f"  - Unanswerable: {len(unanswerable_items)}")
    print(f"JSON saved to:   {args.output_json}")
    print(f"Review saved to: {args.output_review}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

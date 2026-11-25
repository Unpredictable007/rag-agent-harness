"""RAG Failure Analysis Pipeline and Diagnostic Report Generator.

Loads ablation evaluation records (runs/ablation_results.json), the frozen golden
evaluation dataset (eval/golden_set.json), single-shot generation trials
(runs/transcripts/rag_dev_trials.jsonl), and optionally the serialized chunk lookup
(data/bm25_index.pkl) to produce deterministic failure counts, taxonomy classifications,
five-level diagnostic cascade mappings, and reports/rag_failure_analysis.md.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from eval.schema import GoldenDataset, GoldenItem
from src.config import BM25_INDEX_PATH, REPO_ROOT, RUNS_DIR, TRANSCRIPTS_DIR

FailureCategory = Literal[
    "retrieval miss",
    "ranking miss",
    "chunking cut",
    "model ignored context",
    "judge error",
    "wrong golden label",
]

CascadeLevel = Literal[
    "Ingestion",
    "Retrieval",
    "Rerank",
    "Attention",
    "Generation",
]

FAILURE_CATEGORIES: tuple[FailureCategory, ...] = (
    "retrieval miss",
    "ranking miss",
    "chunking cut",
    "model ignored context",
    "judge error",
    "wrong golden label",
)

CASCADE_LEVELS: tuple[CascadeLevel, ...] = (
    "Ingestion",
    "Retrieval",
    "Rerank",
    "Attention",
    "Generation",
)

DEFAULT_ABLATION_PATH = RUNS_DIR / "ablation_results.json"
DEFAULT_GOLDEN_PATH = REPO_ROOT / "eval" / "golden_set.json"
DEFAULT_TRIALS_PATH = TRANSCRIPTS_DIR / "rag_dev_trials.jsonl"
DEFAULT_OUTPUT_REPORT = REPO_ROOT / "reports" / "rag_failure_analysis.md"

_TITLE_STOPWORDS = frozenset(
    {
        "with",
        "from",
        "that",
        "this",
        "your",
        "into",
        "using",
        "posthog",
        "linking",
        "source",
        "guide",
        "start",
        "here",
        "over",
        "about",
    }
)


class FailureAuditAnnotation(BaseModel):
    """Evidence-verified root cause annotation for a single evaluated record."""

    config_name: Literal["E0", "E1", "E2", "E3"]
    item_id: str
    category: FailureCategory
    cascade_level: CascadeLevel
    stage1_evidence: str
    chunk_evidence: str
    root_cause_summary: str


class CorpusIngestionStats(BaseModel):
    """Corpus-wide chunking and metadata quality statistics computed from ChunkLookup."""

    total_chunks: int
    chunks_missing_title_keywords: int
    chunks_with_unstripped_exports: int
    chunks_with_residual_html: int


class AnalyzedFailureCase(BaseModel):
    """Combined runtime record, golden metadata, and verified failure diagnosis."""

    config_name: str
    item_id: str
    question: str
    question_type: str
    answerable: bool
    gold_urls: list[str] = Field(default_factory=list)
    retrieved_urls: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    hit_at_5: float
    recall_at_5: float
    mrr: float
    correctness_score: float
    faithfulness_score: float
    abstain: bool
    generated_answer: str
    correctness_reasoning: str
    faithfulness_reasoning: str
    dev_trial_abstain_count: Optional[int] = None
    dev_trial_total: Optional[int] = None
    category: FailureCategory
    cascade_level: CascadeLevel
    stage1_evidence: str
    chunk_evidence: str
    root_cause_summary: str


class FailureAnalysisReport(BaseModel):
    """Complete deterministic failure analysis summary and case breakdown."""

    dataset_path: str
    ablation_path: str
    trials_path: str
    ablation_timestamp: str
    total_dev_queries: int
    answerable_dev_queries: int
    config_failure_counts: dict[str, dict[str, int]]
    dev_trials_summary: dict[str, Any]
    e3_category_counts: dict[str, int]
    e3_cascade_counts: dict[str, int]
    cohort_category_counts: dict[str, int]
    cohort_cascade_counts: dict[str, int]
    e3_correctness: Optional[float] = None
    e3_single_chunk_recall_at_5: Optional[float] = None
    e3_multi_chunk_recall_at_5: Optional[float] = None
    corpus_stats: Optional[CorpusIngestionStats] = None
    analyzed_cases: list[AnalyzedFailureCase] = Field(default_factory=list)


# Evidence-verified audit annotations covering all 13 E3 degraded/failing records,
# all 6 E2-only degraded/failing records (plus E2 multi-trial alignment), and
# 3 E0/E1 diagnostic records capturing judge error, wrong golden label, and generation hallucination.
FAILURE_AUDIT_ANNOTATIONS: tuple[FailureAuditAnnotation, ...] = (
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-003",
        category="model ignored context",
        cascade_level="Attention",
        stage1_evidence=(
            "Target chunk product-analytics/analyze-data-ai.mdx#4 ranked #1 in Dense, #2 in BM25, "
            "#1 in Hybrid RRF, and #1 in E3 Cross-Encoder reranking (score 0.4685)."
        ),
        chunk_evidence=(
            "Chunk #4 explicitly lists all 5 tips as bullet points, ending with "
            "'Ask questions, not just commands'. E2 included all 5 tips (correctness 1.0), "
            "whereas E3 emitted only the first 4 bullets and dropped the 5th."
        ),
        root_cause_summary=(
            "Target chunk was at rank #1 in the prompt context, but single-shot generation "
            "omitted the final bullet point from the list."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-004",
        category="ranking miss",
        cascade_level="Rerank",
        stage1_evidence=(
            "Target chunk product-analytics/surfaces/mcp.mdx#5 ranked #3 in Dense (E0 correctness 1.0), "
            "0 matches in BM25 top-50, #8 in Stage-1 Hybrid RRF, and #6 in E3 Cross-Encoder (score 0.0084) - "
            "one slot outside the top-5 cutoff."
        ),
        chunk_evidence=(
            "Chunk #5 ('## Dashboard tools') contains the dashboards-bulk-delete table row "
            "but does not contain the token 'MCP' anywhere in body text; 'MCP' exists only in "
            "heading_path ('Use product analytics over PostHog MCP > Dashboard tools'), which "
            "neither BM25 nor CrossEncoder inspects."
        ),
        root_cause_summary=(
            "Omission of heading_path from BM25 and CrossEncoder inputs caused an exact-match "
            "dense hit (#3) to be diluted in RRF (#8) and ranked #6 just outside top-5."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-007",
        category="retrieval miss",
        cascade_level="Retrieval",
        stage1_evidence=(
            "Target chunk feature-flags/best-practices.mdx#5 did not enter Stage-1 top-50 in Dense, "
            "BM25, or Hybrid RRF (only sibling chunks #1, #0, #17, #14 appeared at ranks 14-42)."
        ),
        chunk_evidence=(
            "Section #5 explains deterministic hashing and input root causes using conversational "
            "wording, whereas feature-flags/troubleshooting.mdx#1/#11 directly match query keywords "
            "('feature flag', 'unexpected', 'user') and crowded out best-practices.mdx#5."
        ),
        root_cause_summary=(
            "Stage-1 dense and lexical retrievers saturated top-50 with troubleshooting chunks, "
            "failing to retrieve the deterministic hashing guidance chunk."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-014",
        category="retrieval miss",
        cascade_level="Retrieval",
        stage1_evidence=(
            "Target chunk session-replay/surfaces/web-app.mdx#6 scored 0 matches in Dense top-50, "
            "#43 in BM25 top-50, and missed Stage-1 Hybrid top-50 entirely (only #3 appeared at #38)."
        ),
        chunk_evidence=(
            "Chunk #6 ('### What you can do with a report') lacks the terms 'self-driving' and "
            "'automatically' in body text ('Self-driving reports' is only in heading_path, while "
            "'Why there is no pull request' is split into #5). Generic self-driving/index.mdx#0 "
            "dominated retrieval and caused an opposite answer ('opens a pull request')."
        ),
        root_cause_summary=(
            "Missing heading_path context and terminology overlap with product-wide self-driving "
            "docs prevented the session-replay report chunk from entering Stage-1 Hybrid top-50."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-021",
        category="ranking miss",
        cascade_level="Rerank",
        stage1_evidence=(
            "Target chunk data-pipelines/sources/cisco-meraki.md#4 ranked #4 in Dense, #4 in BM25, "
            "and #1 in Stage-1 Hybrid RRF (E0/E1/E2 all achieved correctness 1.0), but E3 Cross-Encoder "
            "demoted #4 to rank #10 (score 0.9928) behind #0 (0.9992) and 4 other connector sync docs."
        ),
        chunk_evidence=(
            "Chunk #4 is a Markdown table listing all 7 Meraki tables and sync methods, while #0 "
            "is prose overview lacking the 6 full-refresh table names. Because URL-level recall@5 "
            "matched #0, recall@5 stayed 1.0 despite losing the table chunk."
        ),
        root_cause_summary=(
            "Cross-encoder reranker preferred prose sync-mode descriptions from unrelated connectors "
            "over the structured Markdown table in #4, demoting #4 from #1 to #10."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-022",
        category="ranking miss",
        cascade_level="Rerank",
        stage1_evidence=(
            "Target chunk data-pipelines/sources/mintlify.md#3 ranked #9 in BM25 and #20 in Stage-1 "
            "Hybrid RRF (while #2 ranked #1 in Dense and #1 in Hybrid). E3 Cross-Encoder demoted #2 "
            "to rank #9 (0.9682) and #3 to rank #18 (0.9096), yielding 0 Mintlify chunks in top-5."
        ),
        chunk_evidence=(
            "Chunk #3 ('## Available tables') never mentions the word 'Mintlify' in body text "
            "(it resides solely in heading_path 'Linking Mintlify as a source > Available tables'). "
            "Reranker instead promoted crunchbase.md#4, index.mdx#7, lightspeed-retail.md#4, and shipstation.md#2."
        ),
        root_cause_summary=(
            "Without heading_path prepended to candidate text, the cross-encoder could not bind "
            "the Mintlify table chunk to the entity 'Mintlify' and pushed both Mintlify chunks out of top-5."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-037",
        category="ranking miss",
        cascade_level="Rerank",
        stage1_evidence=(
            "First gold URL early-access-feature-management.mdx#0 ranked #1 in E3, while second gold "
            "URL feature-flags/canary-release.md#4 ranked #1 in BM25, #10 in Stage-1 Hybrid RRF, and "
            "#6 in E3 Cross-Encoder (score 0.0266) - one slot outside top-5 (recall@5=0.5, correctness=1.0)."
        ),
        chunk_evidence=(
            "Cross-encoder filled slots #2-#5 with JS usage and general feature-flags chunks "
            "(which incidentally supplied staged-rollout guidance for correctness=1.0), leaving "
            "canary-release.md#4 at #6 and #1 at #12."
        ),
        root_cause_summary=(
            "Secondary gold document canary-release.md was present in the Stage-1 pool (#10) "
            "but stopped at rank #6 under the k=5 rerank cutoff."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-038",
        category="retrieval miss",
        cascade_level="Retrieval",
        stage1_evidence=(
            "Neither experiments/testing-and-launching.mdx#0 nor experiments/sample-size-running-time.mdx#0 "
            "entered Stage-1 Hybrid top-50 (Dense had only sample-size-running-time.mdx#4 at #43; "
            "BM25 and Hybrid had 0 gold URL matches in top-50)."
        ),
        chunk_evidence=(
            "Both #0 chunks begin with unstripped JS 'export const ...' Cloudinary URL strings "
            "(6 lines in sample-size-running-time.mdx#0, ~84% token noise) because clean_mdx() "
            "requires trailing semicolons; testing-and-launching.mdx#0 also omits 'experiment' from its body header."
        ),
        root_cause_summary=(
            "Ingestion export-statement noise and vocabulary mismatch against no-code preview docs "
            "kept both gold documents out of Stage-1 Hybrid top-50."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-039",
        category="chunking cut",
        cascade_level="Ingestion",
        stage1_evidence=(
            "Gold chunk session-replay/start-here.mdx#2 never entered Stage-1 top-50 in Dense, BM25, "
            "or Hybrid; start-here.mdx#0 ranked #7 in E3 Reranker and canvas-recording.mdx#0 ranked #29."
        ),
        chunk_evidence=(
            "session-replay/start-here.mdx uses <QuestLogItem> MDX components without Markdown ## "
            "headings. clean_mdx() stripped <QuestLogItem title=...>, left lowercase <div>/<span> tags, "
            "and split_long_section() severed #2 with orphaned '</span></div>' tags and no pricing heading."
        ),
        root_cause_summary=(
            "JSX component stripping and arbitrary token splitting severed the pricing/billing "
            "section into fragmented HTML-polluted #2, preventing retrieval alongside canvas-recording#0."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-040",
        category="retrieval miss",
        cascade_level="Retrieval",
        stage1_evidence=(
            "Gold URL product-analytics/paths.mdx#0/#2 ranked #1 and #4 in E3, while second gold URL "
            "product-analytics/autocapture.mdx#3 missed Dense, BM25, and Hybrid top-50 completely "
            "(Dense had unrelated autocapture.mdx#18 at #38)."
        ),
        chunk_evidence=(
            "The multi-part query about Android navigation autocapture and User Paths saturated "
            "Stage-1 top-50 with paths.mdx and react-native/index.mdx chunks, starving the short "
            "Android bullet inside autocapture.mdx#3 ('## Supported SDKs')."
        ),
        root_cause_summary=(
            "Single-pass query retrieval over-indexed on User Paths and React Native screens, "
            "failing to surface the Android autocapture defaults chunk into Stage-1 top-50."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-041",
        category="ranking miss",
        cascade_level="Rerank",
        stage1_evidence=(
            "First gold URL web-analytics/surfaces/web-app.mdx#0 ranked #1 in E3, while second gold URL "
            "web-analytics/dashboard.mdx#0 ranked #13 in BM25, #23 in Hybrid RRF, and #16 in E3 "
            "Cross-Encoder (score 0.5758), leaving recall@5=0.5 (correctness=1.0 via surfaces/web-app#0)."
        ),
        chunk_evidence=(
            "Reranker promoted noisy JSX card markup in web-analytics/start-here.mdx#1 (rank #2) and "
            "start-here.mdx#0 (rank #3) above dashboard.mdx#0."
        ),
        root_cause_summary=(
            "Cross-encoder ranked generic web-analytics overview and JSX card chunks ahead of "
            "dashboard.mdx#0, keeping URL recall@5 at 0.5."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-043",
        category="retrieval miss",
        cascade_level="Retrieval",
        stage1_evidence=(
            "Second gold URL error-tracking/upload-mappings/android.mdx#1 ranked #5 in E3, while "
            "first gold URL error-tracking/upload-source-maps/react.mdx#0 never entered Dense, BM25, "
            "or Hybrid top-50."
        ),
        chunk_evidence=(
            "Dual-platform query ('React web frontend built with Vite' + 'Android app with NDK') "
            "pulled react-native.mdx (#1, #2) and android/flutter source-map chunks into top-50; "
            "react.mdx also lost 6 <_snippets> imports stripped by clean_mdx()."
        ),
        root_cause_summary=(
            "Without multi-part query decomposition, React Native and Android tokens crowded out "
            "error-tracking/upload-source-maps/react.mdx#0 from Stage-1 top-50."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E3",
        item_id="gold-044",
        category="retrieval miss",
        cascade_level="Retrieval",
        stage1_evidence=(
            "First gold URL data-warehouse/index.mdx#0 ranked #1 in E3, while second gold URL "
            "data-warehouse/sql/data-access.mdx#1 missed Dense, BM25, and Hybrid top-50 entirely."
        ),
        chunk_evidence=(
            "Query phrases 'amount column comes through as a string' and 'arithmetic' failed to match "
            "data-access.mdx#1 wording ('cast a string property', 'type conversion functions, such as`toString`, "
            "`toFloat`'), while revenue-analytics Stripe docs filled Stage-1 slots."
        ),
        root_cause_summary=(
            "Lexical and semantic vocabulary gap around SQL casting ('arithmetic' vs 'type conversion') "
            "left data-warehouse/sql/data-access.mdx#1 absent from Stage-1 top-50."
        ),
    ),
    # Key E2 failures (also verified across rag_dev_trials.jsonl) that resolved or shifted in E3
    FailureAuditAnnotation(
        config_name="E2",
        item_id="gold-002",
        category="model ignored context",
        cascade_level="Attention",
        stage1_evidence=(
            "Target chunk product-analytics/dashboards.mdx#20 was at rank #1 in E2 Hybrid RRF "
            "(hit@5=1.0, recall@5=1.0, MRR=1.0)."
        ),
        chunk_evidence=(
            "Chunk #20 explicitly states 'This opens a confirmation dialog that lists any other dashboards "
            "using the insight before you confirm. Deletion can be undone after confirming.' E2 omitted "
            "both sentences (correctness 0.5), whereas E3 retained the confirmation dialog (correctness 1.0)."
        ),
        root_cause_summary=(
            "Even with the exact gold chunk at rank #1, single-shot generation dropped two secondary "
            "safety details present at the bottom of the chunk."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E2",
        item_id="gold-009",
        category="model ignored context",
        cascade_level="Attention",
        stage1_evidence=(
            "Target chunk experiments/exposures.mdx#16 was at rank #1 in E2 Hybrid RRF "
            "(hit@5=1.0, recall@5=1.0, MRR=1.0)."
        ),
        chunk_evidence=(
            "Chunk #16 lists 5 numbered best practices; E2 included items 1, 2, 4, and 5 while "
            "skipping item 3 ('Monitor exposure balance'). E3 retrieved #14 (SRM detection) alongside "
            "#16 and scored 1.0."
        ),
        root_cause_summary=(
            "Generator skipped the middle list item (#3 exposure balance) inside the rank-#1 chunk "
            "during synthesis."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E2",
        item_id="gold-012",
        category="chunking cut",
        cascade_level="Ingestion",
        stage1_evidence=(
            "E2 retrieved session-replay/how-to-control-which-sessions-you-record.mdx#1, #10, #7, #3 "
            "in top-5 (URL recall@5=1.0) while #8 ('### Minimum duration') sat outside top-5 at Hybrid #14."
        ),
        chunk_evidence=(
            "chunk_document() split '## Trigger groups' on every '###' subheading into 10 chunks, "
            "separating '### Sampling' (#3) from '### Minimum duration' (#8). golden_set.json also "
            "tagged #2 ('### Controls available', a 6-row link table lacking values) as single-chunk."
        ),
        root_cause_summary=(
            "Aggressive ### subheading splitting separated co-configured trigger-group controls across "
            "distant chunks, causing E2 to answer sampling and fail minimum duration."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E2",
        item_id="gold-015",
        category="ranking miss",
        cascade_level="Rerank",
        stage1_evidence=(
            "Target chunk session-replay/how-to-watch-recordings.mdx#10 ranked #1 in BM25 (E1) and "
            "#14 in Stage-1 Hybrid RRF, missing E2 top-5 and causing 3/3 false abstains in "
            "rag_dev_trials.jsonl (recovered to rank #1 and correctness 1.0 in E3)."
        ),
        chunk_evidence=(
            "Generic 'rage clicks' and 'sessions' chunks from data-pipelines/sources/posthog.mdx#13, "
            "autocapture.mdx#11, and mcp.mdx#2/#3 diluted RRF top-5 above #10."
        ),
        root_cause_summary=(
            "RRF dilution pushed the exact built-in collections table (#10) to rank #14, triggering "
            "consistent false abstains across E2 ablation and all 3 dev trials."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E2",
        item_id="gold-042",
        category="ranking miss",
        cascade_level="Rerank",
        stage1_evidence=(
            "First gold chunk surveys/destinations.mdx#1 ranked #4 in E2, while second gold chunk "
            "surveys/start-here.mdx#1 ranked #13 in BM25 and #29 in Stage-1 Hybrid RRF "
            "(recovered to rank #4 and correctness 1.0 in E3)."
        ),
        chunk_evidence=(
            "start-here.mdx#1 contains the exact Self-driving survey scout answer; #2 was severed "
            "mid-<QuestLogItem>. In E2 top-5, unrelated Slack/scouts chunks occupied ranks #2-#3 and #5."
        ),
        root_cause_summary=(
            "Stage-1 Hybrid RRF buried surveys/start-here.mdx#1 at rank #29 behind generic Slack "
            "and Self-driving chunks, leaving E2 with only half the answer."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E2",
        item_id="gold-045",
        category="ranking miss",
        cascade_level="Rerank",
        stage1_evidence=(
            "First gold chunk data-pipelines/sources/slash.md#1 ranked #1 in E2, while second gold "
            "chunk data-pipelines/sources/postmark.md#1 ranked #2 in BM25 and #6 in Stage-1 Hybrid RRF - "
            "one slot outside top-5 (recovered to rank #2 and correctness 1.0 in E3)."
        ),
        chunk_evidence=(
            "Dense search missed postmark.md#1 completely (0 matches in Dense top-50), so RRF "
            "ranked it #6 behind generic API-key and group-targeting chunks despite BM25 rank #2."
        ),
        root_cause_summary=(
            "Dense miss on Postmark schema terminology pulled BM25 rank-#2 postmark.md#1 down to "
            "Hybrid RRF rank #6, one slot outside the top-5 window."
        ),
    ),
    # Cross-ablation E0/E1 diagnostic cases exposing judge error, wrong golden label, and generation hallucination
    FailureAuditAnnotation(
        config_name="E1",
        item_id="gold-024",
        category="judge error",
        cascade_level="Generation",
        stage1_evidence=(
            "Target chunk data-warehouse/data-quality/troubleshooting.mdx#6 ranked #3 in E1 BM25 "
            "(hit@5=1.0, recall@5=1.0)."
        ),
        chunk_evidence=(
            "Candidate answered both parts asked for the events table accurately ('open error details first', "
            "'Only check recent rows filters on timestamp', 'errored check does not count as failing assertion'). "
            "CorrectnessJudge explicitly hedged ('though it could be 1.0') and docked to 0.5 for omitting "
            "persons/groups created_at and unasked materialization-gate details."
        ),
        root_cause_summary=(
            "Calibrated CorrectnessJudge over-penalized a complete answer by enforcing secondary "
            "reference sentences (persons/groups and materialization gates) not asked in the query."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E0",
        item_id="gold-025",
        category="wrong golden label",
        cascade_level="Ingestion",
        stage1_evidence=(
            "E0 retrieved health-checks/materialized-view-failures.mdx#0 and #3 at ranks #1 and #4, "
            "yielding recall@5=0.0 against gold_urls yet correctness=1.0 and faithfulness=1.0."
        ),
        chunk_evidence=(
            "golden_set.json lists only https://posthog.com/docs/data-warehouse/surfaces/web-app as gold_url, "
            "but health-checks/materialized-view-failures.mdx#3 documents the identical in-app SQL fix "
            "steps in Data modeling without opening a repository PR."
        ),
        root_cause_summary=(
            "Overly narrow gold_urls list omitted an equivalent canonical documentation page "
            "(health-checks/materialized-view-failures), causing a false-zero retrieval score."
        ),
    ),
    FailureAuditAnnotation(
        config_name="E0",
        item_id="gold-038",
        category="retrieval miss",
        cascade_level="Generation",
        stage1_evidence=(
            "E0 Dense failed to retrieve either testing-and-launching.mdx#0 or "
            "sample-size-running-time.mdx#0 in top-5 (hit@5=0.0, recall@5=0.0) and produced "
            "the only faithfulness_score=0.0 record in all 180 ablation evaluations."
        ),
        chunk_evidence=(
            "Missing testing-and-launching.mdx#0, the generator extrapolated from surfaces/web-app.mdx#2 "
            "by instructing the user to confirm logged-out behavior 'using the variant override workflow' - "
            "a claim unsupported by retrieved chunks."
        ),
        root_cause_summary=(
            "Stage-1 retrieval miss combined with over-extrapolation during generation produced an "
            "ungrounded recommendation (faithfulness 0.0) for logged-out variant overrides."
        ),
    ),
)


def sanitize_text(text: str) -> str:
    """Normalize em and en dashes to standard ASCII hyphens."""
    return text.replace("\u2014", "-").replace("\u2013", "-")


def load_ablation_results(path: Path = DEFAULT_ABLATION_PATH) -> dict[str, Any]:
    """Load ablation evaluation results JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Ablation results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dev_trials(path: Path = DEFAULT_TRIALS_PATH) -> list[dict[str, Any]]:
    """Load single-shot generation trials JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Generation trials JSONL file not found: {path}")
    trials: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                trials.append(json.loads(stripped))
    return trials


def is_record_degraded(rec: dict[str, Any]) -> bool:
    """Return True if an evaluation record has imperfect correctness, faithfulness, or answerable recall/abstain."""
    answerable = bool(rec.get("answerable", False))
    corr = float(rec.get("correctness_score", 0.0))
    faith = float(rec.get("faithfulness_score", 0.0))
    rec5 = float(rec.get("recall_at_5", 0.0))
    abstain = bool(rec.get("abstain", False))
    return corr < 1.0 or faith < 1.0 or (answerable and (rec5 < 1.0 or abstain))


def identify_degraded_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records with degraded correctness, faithfulness, recall@5, or false abstention."""
    return [rec for rec in records if is_record_degraded(rec)]


def compute_corpus_ingestion_stats(bm25_path: Path = BM25_INDEX_PATH) -> Optional[CorpusIngestionStats]:
    """Compute ingestion noise and missing title-keyword rates directly from ChunkLookup."""
    if not bm25_path.exists():
        return None
    with open(bm25_path, "rb") as f:
        _, chunk_lookup = pickle.load(f)

    total = len(chunk_lookup)
    missing_title_kw = 0
    unstripped_exports = 0
    residual_html = 0

    for chunk in chunk_lookup:
        text = str(chunk.text)
        text_lower = text.lower()
        if re.search(r"^\s*export\s+(?:const|let|var|function|default)\b", text, re.MULTILINE):
            unstripped_exports += 1
        if re.search(r"</?(?:div|span|p|a|svg|img|ul|li|table|tr|td)\b", text):
            residual_html += 1

        doc_title = str(chunk.heading_path).split(" > ")[0].lower()
        title_words = [
            w for w in re.findall(r"[a-z0-9]{4,}", doc_title) if w not in _TITLE_STOPWORDS
        ]
        if title_words and not any(w in text_lower for w in title_words):
            missing_title_kw += 1

    return CorpusIngestionStats(
        total_chunks=total,
        chunks_missing_title_keywords=missing_title_kw,
        chunks_with_unstripped_exports=unstripped_exports,
        chunks_with_residual_html=residual_html,
    )


def summarize_dev_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize abstain and parse/citation consistency across E2 dev generation trials."""
    by_question: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        by_question.setdefault(str(trial["question_id"]), []).append(trial)

    answerable_trials = [t for t in trials if t.get("answerable", False)]
    false_abstain_trials = [t for t in answerable_trials if t.get("abstain", False)]

    abstain_by_item: dict[str, tuple[int, int]] = {}
    for qid, q_trials in sorted(by_question.items()):
        abst_count = sum(1 for t in q_trials if t.get("abstain", False))
        abstain_by_item[qid] = (abst_count, len(q_trials))

    false_abstain_items = sorted(
        qid
        for qid, q_trials in by_question.items()
        if any(t.get("answerable", False) and t.get("abstain", False) for t in q_trials)
    )
    false_abstain_item_rates = {
        qid: f"{abstain_by_item[qid][0]}/{abstain_by_item[qid][1]}"
        for qid in false_abstain_items
    }

    return {
        "total_trials": len(trials),
        "unique_questions": len(by_question),
        "answerable_trials": len(answerable_trials),
        "false_abstain_trials": len(false_abstain_trials),
        "false_abstain_items": false_abstain_items,
        "false_abstain_item_rates": false_abstain_item_rates,
        "abstain_by_item": abstain_by_item,
    }


def _count_labels(
    cases: list[AnalyzedFailureCase],
    keys: tuple[str, ...],
    attr: str,
) -> dict[str, int]:
    counter = Counter(getattr(c, attr) for c in cases)
    return {k: int(counter.get(k, 0)) for k in keys}


def _relative_display_path(path: Path) -> str:
    """Format a path relative to REPO_ROOT when inside the repository."""
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _summarize_config_failures(
    raw_records: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, int]]]:
    """Index evaluation records by config/item_id and compute per-config failure counts."""
    records_by_cfg: dict[str, dict[str, dict[str, Any]]] = {}
    config_failure_counts: dict[str, dict[str, int]] = {}
    for cfg_name, recs in raw_records.items():
        records_by_cfg[cfg_name] = {str(r["item_id"]): r for r in recs}
        degraded = identify_degraded_records(recs)
        config_failure_counts[cfg_name] = {
            "total_records": len(recs),
            "degraded_or_failing": len(degraded),
            "correctness_lt_1": sum(1 for r in recs if float(r["correctness_score"]) < 1.0),
            "recall_lt_1": sum(
                1 for r in recs if r.get("answerable", False) and float(r["recall_at_5"]) < 1.0
            ),
            "false_abstain": sum(
                1 for r in recs if r.get("answerable", False) and bool(r.get("abstain", False))
            ),
            "faithfulness_lt_1": sum(1 for r in recs if float(r["faithfulness_score"]) < 1.0),
        }
    return records_by_cfg, config_failure_counts


def _build_analyzed_cases(
    annotations: tuple[FailureAuditAnnotation, ...],
    records_by_cfg: dict[str, dict[str, dict[str, Any]]],
    items_by_id: dict[str, GoldenItem],
    abstain_by_item: dict[str, tuple[int, int]],
) -> list[AnalyzedFailureCase]:
    """Validate audit annotations against runtime records and construct AnalyzedFailureCase models."""
    analyzed_cases: list[AnalyzedFailureCase] = []
    seen_keys: set[tuple[str, str]] = set()

    for ann in annotations:
        key = (ann.config_name, ann.item_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate audit annotation for {key}")
        seen_keys.add(key)

        cfg_records = records_by_cfg.get(ann.config_name)
        if cfg_records is None or ann.item_id not in cfg_records:
            raise KeyError(f"Annotated record {key} not found in ablation_results.json")
        if ann.item_id not in items_by_id:
            raise KeyError(f"Annotated item '{ann.item_id}' not found in golden_set.json")

        rec = cfg_records[ann.item_id]
        if not is_record_degraded(rec):
            raise ValueError(f"Annotated record {key} is not degraded or failing")

        golden_item = items_by_id[ann.item_id]
        trial_abst, trial_tot = abstain_by_item.get(ann.item_id, (0, 0))
        analyzed_cases.append(
            AnalyzedFailureCase(
                config_name=ann.config_name,
                item_id=ann.item_id,
                question=sanitize_text(golden_item.question),
                question_type=str(golden_item.type),
                answerable=bool(golden_item.answerable),
                gold_urls=list(golden_item.gold_urls),
                retrieved_urls=list(rec.get("retrieved_urls", [])),
                retrieved_chunk_ids=list(rec.get("retrieved_chunk_ids", [])),
                hit_at_5=float(rec["hit_at_5"]),
                recall_at_5=float(rec["recall_at_5"]),
                mrr=float(rec["mrr"]),
                correctness_score=float(rec["correctness_score"]),
                faithfulness_score=float(rec["faithfulness_score"]),
                abstain=bool(rec["abstain"]),
                generated_answer=sanitize_text(str(rec.get("generated_answer", ""))),
                correctness_reasoning=sanitize_text(str(rec.get("correctness_reasoning", ""))),
                faithfulness_reasoning=sanitize_text(str(rec.get("faithfulness_reasoning", ""))),
                dev_trial_abstain_count=trial_abst if ann.config_name == "E2" else None,
                dev_trial_total=trial_tot if ann.config_name == "E2" else None,
                category=ann.category,
                cascade_level=ann.cascade_level,
                stage1_evidence=sanitize_text(ann.stage1_evidence),
                chunk_evidence=sanitize_text(ann.chunk_evidence),
                root_cause_summary=sanitize_text(ann.root_cause_summary),
            )
        )
    return analyzed_cases


def _extract_e3_summary_metrics(
    ablation: dict[str, Any],
    e3_records: list[dict[str, Any]],
    items_by_id: dict[str, GoldenItem],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract E3 overall correctness and single/multi-chunk Recall@5 from summaries or raw records."""
    e3_correctness: Optional[float] = None
    single_recall: Optional[float] = None
    multi_recall: Optional[float] = None

    for summary in ablation.get("summaries", []):
        if summary.get("config_name") == "E3":
            if "correctness" in summary:
                e3_correctness = float(summary["correctness"])
            by_type = summary.get("by_type", {})
            if "recall_at_5" in by_type.get("single-chunk", {}):
                single_recall = float(by_type["single-chunk"]["recall_at_5"])
            if "recall_at_5" in by_type.get("multi-chunk", {}):
                multi_recall = float(by_type["multi-chunk"]["recall_at_5"])
            break

    if e3_correctness is None and e3_records:
        e3_correctness = sum(float(r["correctness_score"]) for r in e3_records) / len(e3_records)

    def _mean_type_recall(target_type: str) -> Optional[float]:
        matching = [
            float(r["recall_at_5"])
            for r in e3_records
            if (r.get("question_type") or (items_by_id[str(r["item_id"])].type if str(r["item_id"]) in items_by_id else None))
            == target_type
        ]
        return (sum(matching) / len(matching)) if matching else None

    if single_recall is None:
        single_recall = _mean_type_recall("single-chunk")
    if multi_recall is None:
        multi_recall = _mean_type_recall("multi-chunk")

    return e3_correctness, single_recall, multi_recall


def analyze_failures(
    ablation_path: Path = DEFAULT_ABLATION_PATH,
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    trials_path: Path = DEFAULT_TRIALS_PATH,
    bm25_path: Path = BM25_INDEX_PATH,
    annotations: tuple[FailureAuditAnnotation, ...] = FAILURE_AUDIT_ANNOTATIONS,
    include_corpus_stats: bool = True,
) -> FailureAnalysisReport:
    """Run deterministic failure analysis over ablation, golden, and generation trial artifacts."""
    ablation = load_ablation_results(ablation_path)
    dataset = GoldenDataset.from_file(golden_path)
    items_by_id: dict[str, GoldenItem] = {it.id: it for it in dataset.items}
    trials = load_dev_trials(trials_path)
    trials_summary = summarize_dev_trials(trials)
    abstain_by_item: dict[str, tuple[int, int]] = trials_summary["abstain_by_item"]

    raw_records = ablation.get("records", {})
    if "E3" not in raw_records:
        raise KeyError("Expected 'E3' records in ablation_results.json")
    records_by_cfg, config_failure_counts = _summarize_config_failures(raw_records)

    e3_degraded_ids = {str(r["item_id"]) for r in identify_degraded_records(raw_records["E3"])}
    annotated_e3_ids = {ann.item_id for ann in annotations if ann.config_name == "E3"}
    missing_e3 = sorted(e3_degraded_ids - annotated_e3_ids)
    if missing_e3:
        raise ValueError(f"Missing audit annotations for E3 degraded items: {missing_e3}")

    analyzed_cases = _build_analyzed_cases(
        annotations=annotations,
        records_by_cfg=records_by_cfg,
        items_by_id=items_by_id,
        abstain_by_item=abstain_by_item,
    )

    e3_cases = [c for c in analyzed_cases if c.config_name == "E3"]
    e3_category_counts = _count_labels(e3_cases, FAILURE_CATEGORIES, "category")
    e3_cascade_counts = _count_labels(e3_cases, CASCADE_LEVELS, "cascade_level")
    cohort_category_counts = _count_labels(analyzed_cases, FAILURE_CATEGORIES, "category")
    cohort_cascade_counts = _count_labels(analyzed_cases, CASCADE_LEVELS, "cascade_level")

    e3_correctness, e3_single_recall, e3_multi_recall = _extract_e3_summary_metrics(
        ablation=ablation,
        e3_records=raw_records["E3"],
        items_by_id=items_by_id,
    )
    corpus_stats = compute_corpus_ingestion_stats(bm25_path) if include_corpus_stats else None

    dev_items = dataset.filter_by_split("dev")
    answerable_dev = [it for it in dev_items if it.answerable]

    return FailureAnalysisReport(
        dataset_path=_relative_display_path(golden_path),
        ablation_path=_relative_display_path(ablation_path),
        trials_path=_relative_display_path(trials_path),
        ablation_timestamp=str(ablation.get("timestamp", "unknown")),
        total_dev_queries=len(dev_items),
        answerable_dev_queries=len(answerable_dev),
        config_failure_counts=config_failure_counts,
        dev_trials_summary={
            k: v for k, v in trials_summary.items() if k != "abstain_by_item"
        },
        e3_category_counts=e3_category_counts,
        e3_cascade_counts=e3_cascade_counts,
        cohort_category_counts=cohort_category_counts,
        cohort_cascade_counts=cohort_cascade_counts,
        e3_correctness=e3_correctness,
        e3_single_chunk_recall_at_5=e3_single_recall,
        e3_multi_chunk_recall_at_5=e3_multi_recall,
        corpus_stats=corpus_stats,
        analyzed_cases=analyzed_cases,
    )


def _format_pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(100.0 * count / total):.1f}%"


def _truncate_text(text: str, limit: int = 340) -> str:
    """Truncate text to at most `limit` characters, snapping cleanly to a word boundary."""
    clean = sanitize_text(" ".join(text.split()))
    if len(clean) <= limit:
        return clean
    if limit <= 3:
        return clean[:limit]
    cutoff = limit - 3
    prefix = clean[:cutoff]
    if clean[cutoff] == " ":
        snapped = prefix.rstrip()
    elif " " in prefix:
        snapped = prefix.rsplit(" ", 1)[0].rstrip()
    else:
        snapped = prefix
    return (snapped or prefix) + "..."


def _format_trial_stability_line(dev_trials_summary: dict[str, Any]) -> str:
    """Format the multi-trial false-abstain summary line dynamically."""
    false_trials = int(dev_trials_summary.get("false_abstain_trials", 0))
    ans_trials = int(dev_trials_summary.get("answerable_trials", 0))
    false_items: list[str] = list(dev_trials_summary.get("false_abstain_items", []))
    item_rates: dict[str, str] = dict(dev_trials_summary.get("false_abstain_item_rates", {}))
    pct = _format_pct(false_trials, ans_trials)

    if not false_items:
        return (
            f"- **E2 Multi-Trial Stability (`rag_dev_trials.jsonl`):** `{false_trials}` / "
            f"`{ans_trials}` answerable trials (`{pct}`) produced false abstains across `0` items."
        )

    rates = [item_rates.get(item, "") for item in false_items]
    unique_rates = {r for r in rates if r}
    if len(unique_rates) == 1:
        rate_note = f" (all {len(false_items)} abstained in {next(iter(unique_rates))} trials)"
    elif any(rates):
        rate_note = " (" + ", ".join(f"{it}: {item_rates[it]}" for it in false_items if it in item_rates) + ")"
    else:
        rate_note = ""

    items_str = ", ".join(f"`{x}`" for x in false_items)
    return (
        f"- **E2 Multi-Trial Stability (`rag_dev_trials.jsonl`):** `{false_trials}` / "
        f"`{ans_trials}` answerable trials (`{pct}`) produced deterministic false "
        f"abstains across `{len(false_items)}` items: {items_str}{rate_note}."
    )


def _render_executive_summary(
    report: FailureAnalysisReport,
    e3_total: int,
    e2_total: int,
    other_total: int,
) -> list[str]:
    """Render header metadata and Section 1 (Executive Summary)."""
    cohort_total = len(report.analyzed_cases)
    lines: list[str] = [
        "# RAG Failure Analysis & Diagnostic Report",
        "",
        f"- **Ablation Timestamp:** `{report.ablation_timestamp}`",
        f"- **Dataset:** `{report.dataset_path}` (`dev` split: `{report.total_dev_queries}` total, `{report.answerable_dev_queries}` answerable)",
        f"- **Ablation Records Source:** `{report.ablation_path}`",
        f"- **Multi-Trial Generation Source:** `{report.trials_path}` (`{report.dev_trials_summary['total_trials']}` E2 trials)",
        f"- **Inspected Failure Cohort:** `{cohort_total}` failing/degraded records (`{e3_total}` in production baseline `E3`, `{cohort_total - e3_total}` in `E2`/`E1`/`E0`)",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        (
            "End-to-end evaluation of the production RAG configuration (`E3`: Hybrid RRF + `BAAI/bge-reranker-base`) "
            f"across `{report.total_dev_queries}` dev queries shows `{e3_total}` degraded or failing records "
            "(correctness < 1.0, faithfulness < 1.0, or answerable recall@5 < 1.0 / false abstain). "
            f"Combining these `{e3_total}` E3 cases with `{e2_total}` additional `E2` Hybrid RRF failures (verified across "
            f"`{report.dev_trials_summary['total_trials']}` single-shot trials) and `{other_total}` cross-ablation `E0`/`E1` diagnostic "
            f"cases yields `{cohort_total}` evidence-audited failure records spanning all six taxonomy categories and all "
            "five stages of the diagnostic cascade."
        ),
        "",
        f"### Cross-Configuration Failure Footprint (`dev` split, n={report.total_dev_queries})",
        "",
        "| Configuration | Degraded / Failing Records | Correctness < 1.0 | Answerable Recall@5 < 1.0 | False Abstains | Faithfulness < 1.0 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cfg in ("E0", "E1", "E2", "E3"):
        c = report.config_failure_counts.get(cfg, {})
        lines.append(
            f"| **{cfg}** | {c.get('degraded_or_failing', 0)} / {c.get('total_records', 0)} | "
            f"{c.get('correctness_lt_1', 0)} | {c.get('recall_lt_1', 0)} | "
            f"{c.get('false_abstain', 0)} | {c.get('faithfulness_lt_1', 0)} |"
        )

    lines.extend(["", _format_trial_stability_line(report.dev_trials_summary)])

    if report.corpus_stats is not None:
        cs = report.corpus_stats
        lines.append(
            f"- **Corpus Ingestion Audit (`data/bm25_index.pkl`, n={cs.total_chunks:,} chunks):** "
            f"`{cs.chunks_missing_title_keywords:,}` chunks (`{_format_pct(cs.chunks_missing_title_keywords, cs.total_chunks)}`) "
            "lack document-title topic keywords in `chunk.text` (stored only in `heading_path`), "
            f"`{cs.chunks_with_unstripped_exports:,}` chunks (`{_format_pct(cs.chunks_with_unstripped_exports, cs.total_chunks)}`) "
            "contain unstripped JS `export` statements, and "
            f"`{cs.chunks_with_residual_html:,}` chunks (`{_format_pct(cs.chunks_with_residual_html, cs.total_chunks)}`) "
            "contain residual HTML layout tags (`<div>`, `<span>`)."
        )

    return lines


def _render_taxonomy_and_cascade_tables(
    report: FailureAnalysisReport,
    e3_total: int,
    cohort_total: int,
) -> list[str]:
    """Render Section 2 (Failure Category Counts) and Section 3 (Five-Level Cascade)."""
    category_descriptions = {
        "retrieval miss": "Target chunk absent from Stage-1 top-50 candidate pool.",
        "ranking miss": "Target chunk present in Stage-1/other-mode candidates but ranked outside top-5.",
        "chunking cut": "Answer severed across chunk boundaries or broken by MDX/HTML truncation.",
        "model ignored context": "Target chunk present in top-5 prompt context, yet LLM omitted key facts.",
        "judge error": "LLM judge over-penalized a complete answer on unasked reference details.",
        "wrong golden label": "Gold URL set omitted an equivalent canonical documentation page.",
    }
    cascade_descriptions = {
        "Ingestion": "MDX `export`/HTML pollution, `<QuestLogItem>` splits, `###` fragmentation, narrow gold URLs.",
        "Retrieval": "Single-query embeddings/BM25 saturated top-50 with dominant topics or missed synonyms.",
        "Rerank": "Cross-encoder (`bge-reranker-base`) or RRF pushed table/heading-dependent chunks below rank #5.",
        "Attention": "Generator dropped 1-2 details or list bullets from a rank-#1 chunk inside `<retrieved_context>`.",
        "Generation": "Judge over-penalization on secondary details (`E1`) or ungrounded extrapolation (`E0`).",
    }

    lines: list[str] = [
        "",
        "---",
        "",
        "## 2. Failure Category Counts (6-Category Taxonomy)",
        "",
        f"| Failure Category | E3 Baseline (n={e3_total}) | E3 Share | Full Inspected Cohort (n={cohort_total}) | Cohort Share | Primary Diagnostic Mechanism |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for cat in FAILURE_CATEGORIES:
        e3_cnt = report.e3_category_counts[cat]
        coh_cnt = report.cohort_category_counts[cat]
        lines.append(
            f"| `{cat}` | {e3_cnt} | {_format_pct(e3_cnt, e3_total)} | "
            f"{coh_cnt} | {_format_pct(coh_cnt, cohort_total)} | {category_descriptions[cat]} |"
        )
    lines.extend(
        [
            f"| **Total** | **{e3_total}** | **{_format_pct(e3_total, e3_total)}** | **{cohort_total}** | **{_format_pct(cohort_total, cohort_total)}** | - |",
            "",
            "---",
            "",
            "## 3. Five-Level Diagnostic Cascade (`Ingestion -> Retrieval -> Rerank -> Attention -> Generation`)",
            "",
            f"| Cascade Level | E3 Baseline (n={e3_total}) | E3 Share | Full Inspected Cohort (n={cohort_total}) | Cohort Share | Observed Failure Signature |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for lvl in CASCADE_LEVELS:
        e3_cnt = report.e3_cascade_counts[lvl]
        coh_cnt = report.cohort_cascade_counts[lvl]
        lines.append(
            f"| **{lvl}** | {e3_cnt} | {_format_pct(e3_cnt, e3_total)} | "
            f"{coh_cnt} | {_format_pct(coh_cnt, cohort_total)} | {cascade_descriptions[lvl]} |"
        )
    lines.append(
        f"| **Total** | **{e3_total}** | **{_format_pct(e3_total, e3_total)}** | **{cohort_total}** | **{_format_pct(cohort_total, cohort_total)}** | - |"
    )
    return lines


def _render_case_sections(
    e3_cases: list[AnalyzedFailureCase],
    e2_cases: list[AnalyzedFailureCase],
    other_cases: list[AnalyzedFailureCase],
) -> list[str]:
    """Render Section 4 (Detailed Case-by-Case Diagnostic Analysis)."""
    lines: list[str] = [
        "",
        "---",
        "",
        "## 4. Detailed Case-by-Case Diagnostic Analysis",
        "",
    ]
    sections = [
        (f"4.1 Production Baseline (`E3`) Degraded & Failing Cases (n={len(e3_cases)})", e3_cases),
        (f"4.2 Hybrid RRF (`E2`) & Dev-Trial Failures (n={len(e2_cases)})", e2_cases),
        (f"4.3 Cross-Ablation (`E0` / `E1`) Diagnostic Edge Cases (n={len(other_cases)})", other_cases),
    ]

    for sec_title, cases in sections:
        lines.extend([f"### {sec_title}", ""])
        for case in cases:
            trial_note = ""
            if case.dev_trial_total:
                trial_note = f" | Dev Trials Abstain: `{case.dev_trial_abstain_count}/{case.dev_trial_total}`"
            lines.append(
                f"#### `[{case.config_name}] {case.item_id}` ({case.question_type}) - `{case.category}` | Cascade: `{case.cascade_level}`"
            )
            lines.append(f"- **Question:** {case.question}")
            lines.append(
                f"- **Metrics:** Hit@5=`{case.hit_at_5:.2f}` | Recall@5=`{case.recall_at_5:.2f}` | MRR=`{case.mrr:.2f}` | "
                f"Correctness=`{case.correctness_score:.1f}` | Faithfulness=`{case.faithfulness_score:.1f}` | "
                f"Abstain=`{case.abstain}`{trial_note}"
            )
            lines.append("- **Gold URLs:** " + (", ".join(f"`{u}`" for u in case.gold_urls) or "None"))
            lines.append("- **Retrieved URLs (Top-5):** " + (", ".join(f"`{u}`" for u in case.retrieved_urls) or "None"))
            lines.append("- **Retrieved Chunk IDs:** " + (", ".join(f"`{cid}`" for cid in case.retrieved_chunk_ids) or "None"))
            lines.append(f"- **Judge Reasoning (Correctness):** {_truncate_text(case.correctness_reasoning)}")
            if case.faithfulness_score < 1.0:
                lines.append(f"- **Judge Reasoning (Faithfulness):** {_truncate_text(case.faithfulness_reasoning)}")
            lines.append(f"- **Stage-1 & Retrieval Evidence:** {case.stage1_evidence}")
            lines.append(f"- **Chunk & Doc Evidence:** {case.chunk_evidence}")
            lines.append(f"- **Verified Root Cause:** {case.root_cause_summary}")
            lines.append("")
    return lines


def _render_hypotheses(report: FailureAnalysisReport) -> list[str]:
    """Render Section 5 (Prioritized Fix Hypotheses) dynamically from report data."""
    e3_cases = [c for c in report.analyzed_cases if c.config_name == "E3"]
    e2_cases = [c for c in report.analyzed_cases if c.config_name == "E2"]
    e3_total = sum(report.e3_category_counts.values()) or len(e3_cases)

    hyp1_cases = [
        c
        for c in e3_cases
        if any(
            kw in f"{c.chunk_evidence} {c.root_cause_summary}".lower()
            for kw in ("heading_path", "export", "<questlogitem>", "jsx", "markdown table")
        )
    ]
    hyp1_ids = ", ".join(f"`{c.item_id}`" for c in hyp1_cases) or "None"
    hyp1_count = len(hyp1_cases)

    if report.corpus_stats is not None:
        cs = report.corpus_stats
        missing_pct = _format_pct(cs.chunks_missing_title_keywords, cs.total_chunks)
        corpus_clause = (
            f"leaving `{missing_pct}` of all corpus chunks (`{cs.chunks_missing_title_keywords:,}/{cs.total_chunks:,}`) "
            "without their document-title topic keywords in `chunk.text` (e.g., `'MCP'` absent from `mcp.mdx#5`; "
            "`'Mintlify'` absent from `mintlify.md#3`). Similarly, `src/retrieval.py` feeds only `cand.text` to "
            "`BAAI/bge-reranker-base`. Meanwhile, `clean_mdx()` leaves semicolon-less `export const` URL lines "
            f"(`{cs.chunks_with_unstripped_exports:,}` chunks) and `<QuestLogItem>` / HTML tags "
            f"(`{cs.chunks_with_residual_html:,}` chunks) intact."
        )
    else:
        corpus_clause = (
            "leaving corpus chunks without their document-title topic keywords in `chunk.text` "
            "(e.g., `'MCP'` absent from `mcp.mdx#5`; `'Mintlify'` absent from `mintlify.md#3`). "
            "Similarly, `src/retrieval.py` feeds only `cand.text` to `BAAI/bge-reranker-base`. "
            "Meanwhile, `clean_mdx()` leaves semicolon-less `export const` URL lines and `<QuestLogItem>` / HTML tags intact."
        )

    hyp2_cases = [
        c
        for c in e3_cases
        if (
            (c.question_type == "multi-chunk" and c.category == "retrieval miss")
            or (
                c.question_type == "multi-chunk"
                and c.category == "ranking miss"
                and "jsx" not in f"{c.chunk_evidence} {c.root_cause_summary}".lower()
            )
            or (
                c.category == "retrieval miss"
                and "saturated" in f"{c.chunk_evidence} {c.root_cause_summary}".lower()
            )
        )
    ]
    hyp2_ids = ", ".join(f"`{c.item_id}`" for c in hyp2_cases) or "None"
    multi_target_ids = (
        ", ".join(f"`{c.item_id}`" for c in hyp2_cases if c.question_type == "multi-chunk")
        or "multi-chunk queries"
    )
    multi_rec_str = (
        f"`{report.e3_multi_chunk_recall_at_5:.4f}`"
        if report.e3_multi_chunk_recall_at_5 is not None
        else "`N/A`"
    )
    single_rec_str = (
        f"`{report.e3_single_chunk_recall_at_5:.4f}`"
        if report.e3_single_chunk_recall_at_5 is not None
        else "`N/A`"
    )
    e3_corr_str = f"`{report.e3_correctness:.4f}`" if report.e3_correctness is not None else "`N/A`"

    hyp3_parts: list[str] = []
    e3_attn = [c.item_id for c in e3_cases if c.category == "model ignored context"]
    if e3_attn:
        hyp3_parts.append(", ".join(f"`{x}`" for x in e3_attn) + " (E3)")
    e2_attn_chunk = [
        c.item_id for c in e2_cases if c.category in ("model ignored context", "chunking cut")
    ]
    if e2_attn_chunk:
        hyp3_parts.append(" / ".join(f"`{x}`" for x in e2_attn_chunk) + " (E2)")
    for c in report.analyzed_cases:
        if c.category in ("judge error", "wrong golden label"):
            hyp3_parts.append(f"`{c.item_id}` ({c.config_name} `{c.category}`)")
    hyp3_targets = ", ".join(hyp3_parts) or "None"

    return [
        "---",
        "",
        "## 5. Prioritized Fix Hypotheses (Top 3)",
        "",
        "### Hypothesis 1 (Priority 1 - Highest Impact): Context-Aware Chunk Serialization & MDX Ingestion Cleanup",
        f"- **Target Failures:** {hyp1_ids} (`{hyp1_count}/{e3_total}` E3 degraded cases).",
        (
            "- **Root Mechanism:** `src/ingest.py` (`build_indexes`) embeds and BM25-tokenizes `chunk.text` without "
            f"`heading_path` or `title`, {corpus_clause}"
        ),
        (
            "- **Proposed Fix:** (1) Prepend `[Document: {heading_path}]` to chunk text for BM25 indexing, dense embedding, "
            "and cross-encoder reranking (`(query, f\"{cand.heading_path}\\n{cand.text}\")`). (2) Update `clean_mdx()` to strip "
            "semicolon-optional `export` declarations, convert `<QuestLogItem title=...>` / `<Step title=...>` titles into "
            "Markdown subheadings, and strip residual HTML layout tags (`<div>`, `<span>`)."
        ),
        (
            "- **Expected Impact:** Directly resolves `gold-004`, `gold-021`, and `gold-022` reranker demotions and removes "
            "noise blocking `gold-014`, `gold-038`, and `gold-039`, targeting `+0.08` to `+0.14` E3 Recall@5 and `+0.06` to "
            "`+0.11` E3 Correctness on the `dev` split."
        ),
        (
            "- **How to Validate:** Rebuild indexes via `python -m src.ingest`, verify `pytest tests/test_regression_gate.py` "
            "passes with zero regression, and re-run `python -m eval.eval_ablation` on `dev` (keeping `holdout` frozen) to "
            f"confirm gains on the {hyp1_count} target queries."
        ),
        "",
        "### Hypothesis 2 (Priority 2 - High Impact on Multi-Chunk Queries): Multi-Part Query Decomposition & Per-URL Diversification",
        f"- **Target Failures:** {hyp2_ids} (`{len(hyp2_cases)}/{e3_total}` E3 degraded cases; multi-chunk Recall@5 is {multi_rec_str} vs {single_rec_str} single-chunk).",
        (
            "- **Root Mechanism:** Multi-part queries combining two products or workflows (e.g., React+Vite and Android NDK in "
            "`gold-043`; Android autocapture and User Paths in `gold-040`; Stripe SQL joins and casting in `gold-044`) saturate "
            "single-query dense and BM25 top-50 pools with one dominant topic, starving the second required document out of Stage-1."
        ),
        (
            "- **Proposed Fix:** Add lightweight query decomposition for compound questions (generating 2-3 focused sub-queries, "
            "running hybrid retrieval per sub-query, fusing via RRF with `candidate_pool_size=100`) and enforce a soft per-URL "
            "diversity cap (`max_chunks_per_url=2`) in the top-5 reranked output."
        ),
        (
            f"- **Expected Impact:** Elevates secondary-document Stage-1 coverage on multi-chunk queries ({multi_target_ids}), "
            f"projected to lift multi-chunk Recall@5 from {multi_rec_str} toward `0.78-0.89` and overall E3 "
            f"Correctness from {e3_corr_str} above `0.90`."
        ),
        (
            "- **How to Validate:** Benchmark single-query E3 vs decomposed E3 on `eval/golden_set.json` (`dev` split) using "
            "`eval/eval_ablation.py`, checking multi-chunk Recall@5, Correctness, and P95 end-to-end latency within SLA."
        ),
        "",
        "### Hypothesis 3 (Priority 3 - Targeted Completeness & Eval Hygiene): Subsection Merging, List-Completeness Prompting & Judge/Gold Calibration",
        f"- **Target Failures:** {hyp3_targets}.",
        (
            "- **Root Mechanism:** Two distinct micro-defects remain when retrieval succeeds: (a) `chunk_document()` splits "
            "small `###` sibling subsections into isolated chunks (`gold-012` separated `Sampling` `#3` from `Minimum duration` `#8`), "
            "and single-shot generation occasionally drops 1 of 5 bullet points from rank-#1 chunks (`gold-003`, `gold-002`, `gold-009`); "
            "(b) `CorrectnessJudge` occasionally docks answers for omitting unasked reference details (`gold-024`), and `gold_urls` "
            "omits equivalent canonical docs (`gold-025`)."
        ),
        (
            "- **Proposed Fix:** (1) Merge adjacent small `###` subsections under their parent `##` heading up to "
            "`CHUNK_MAX_TOKENS=500` during ingestion. (2) Add a multi-item completeness check in `src/generation.py` `SYSTEM_PROMPT`. "
            "(3) Refine `CorrectnessJudge` instructions to anchor completeness strictly to user-asked clauses, and audit `gold_urls` "
            "for equivalent canonical pages (`health-checks/materialized-view-failures` on `gold-025`; `gold-012` multi-chunk tag)."
        ),
        (
            "- **Expected Impact:** Eliminates `Attention` partial-credit drops (`0.5 -> 1.0` on `gold-003`, `gold-002`, `gold-009`), "
            "keeps co-configured controls intact (`gold-012`), and removes false metric/judge penalties (`gold-024`, `gold-025`)."
        ),
        (
            "- **How to Validate:** Re-run `eval/calibrate_judge.py` against `eval/human_labels_30.json` to preserve kappa = 1.000, "
            "execute 3-trial generation (`eval/run_generation.py`) to confirm zero bullet-omission regressions, and verify "
            "`eval/validate_golden_set.py` passes."
        ),
        "",
    ]


def render_markdown_report(report: FailureAnalysisReport) -> str:
    """Render the complete diagnostic failure analysis report in Markdown."""
    e3_cases = [c for c in report.analyzed_cases if c.config_name == "E3"]
    e2_cases = [c for c in report.analyzed_cases if c.config_name == "E2"]
    other_cases = [c for c in report.analyzed_cases if c.config_name in ("E0", "E1")]
    e3_total = sum(report.e3_category_counts.values())
    cohort_total = len(report.analyzed_cases)

    lines: list[str] = []
    lines.extend(_render_executive_summary(report, e3_total, len(e2_cases), len(other_cases)))
    lines.extend(_render_taxonomy_and_cascade_tables(report, e3_total, cohort_total))
    lines.extend(_render_case_sections(e3_cases, e2_cases, other_cases))
    lines.extend(_render_hypotheses(report))
    return sanitize_text("\n".join(lines))


def write_failure_analysis_report(
    report: FailureAnalysisReport,
    output_path: Path = DEFAULT_OUTPUT_REPORT,
) -> Path:
    """Render and write the failure analysis markdown report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown_report(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
        if not markdown.endswith("\n"):
            f.write("\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RAG ablation and generation failures.")
    parser.add_argument(
        "--ablation",
        type=Path,
        default=DEFAULT_ABLATION_PATH,
        help="Path to runs/ablation_results.json",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to eval/golden_set.json",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=DEFAULT_TRIALS_PATH,
        help="Path to runs/transcripts/rag_dev_trials.jsonl",
    )
    parser.add_argument(
        "--bm25",
        type=Path,
        default=BM25_INDEX_PATH,
        help="Path to data/bm25_index.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
        help="Output path for markdown diagnostic report",
    )
    args = parser.parse_args()

    report = analyze_failures(
        ablation_path=args.ablation,
        golden_path=args.golden,
        trials_path=args.trials,
        bm25_path=args.bm25,
    )
    out_path = write_failure_analysis_report(report, output_path=args.output)
    print(f"Wrote failure analysis report to {out_path}")
    print(f"E3 category counts (n={sum(report.e3_category_counts.values())}): {report.e3_category_counts}")
    print(f"E3 cascade counts: {report.e3_cascade_counts}")
    print(f"Cohort category counts (n={len(report.analyzed_cases)}): {report.cohort_category_counts}")
    print(f"Cohort cascade counts: {report.cohort_cascade_counts}")


if __name__ == "__main__":
    main()

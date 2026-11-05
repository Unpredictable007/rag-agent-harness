# Generation Evaluation Report: Structured RAG Pipeline (Milestone M5)

- **Generated:** `2026-11-05T11:08:51+00:00`
- **Dataset:** `eval/golden_set.json`
- **Split:** `dev` (45 questions: 27 single-chunk, 9 multi-chunk, 9 unanswerable)
- **Trials per Question:** 3 (135 total evaluation runs)
- **Context Source:** Hybrid RRF retrieval ($k=5$, Dense + BM25, $k_{\text{rrf}}=60$)
- **Primary LLM:** Claude Haiku 5.5 (`claude-haiku-5-5`, reasoning effort: `xhigh`)
- **Fallback LLM:** Gemini 3.8 Flash (`gemini-3.8-flash`, thinking effort: `HIGH`)
- **Transcripts Log:** `runs/transcripts/rag_dev_trials.jsonl`

---

## 1. Executive Summary

Milestone M5 introduces the structured single-shot generation pipeline for technical documentation QA over the PostHog documentation corpus. The pipeline enforces:
1. **Strict Context Isolation:** Candidate chunks are encapsulated in explicit `<retrieved_context>` XML tags and framed as untrusted passive reference data to prevent prompt injection and untrusted instruction following.
2. **Strict Output Contract:** Deterministic JSON schema compliance enforced via Pydantic `GenerationOutput(answer, citations, abstain)`.
3. **Deterministic Citation Verification:** Post-generation programmatic assertion ensuring every cited `chunk_id` belongs to the candidate set provided in the prompt ($C \subseteq S$). Hallucinated chunk IDs immediately invalidate the trial.
4. **Abstain Policy Enforcement:** Strict domain assertion requiring empty citations (`citations: []`) and standard refusal text when the context lacks sufficient evidence.
5. **Configurable Failover Architecture:** Claude Haiku as primary with automatic bidirectional failover to Gemini Flash upon retryable HTTP errors (529, 503, 429, 500).

Evaluation across 135 multi-trial runs on the frozen dev set demonstrates **100.0% parse success, 0.0% hallucinated citations, and 100.0% abstain accuracy on unanswerable questions**.

---

## 2. Benchmark Summary Metrics

| Metric | Target (M5 Spec) | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Parse Success Rate** | $\ge 95.0\%$ | **100.0%** (135/135) | **PASSED** |
| **Strict Valid Output Rate** | High | **100.0%** (135/135) | **PASSED** |
| **Hallucinated Citation Rate** | $0.0\%$ | **0.0%** (0/204 citations) | **PASSED** |
| **Trials with Hallucinations** | $0.0\%$ | **0.0%** (0/135 trials) | **PASSED** |
| **Abstain Accuracy (Unanswerable)** | $100.0\%$ | **100.0%** (27/27 trials) | **PASSED** |
| **False Abstain Rate (Answerable)** | Controlled | **11.1%** (12/108 trials) | **HEALTHY** |
| **P50 Generation Latency** | - | **3170.1 ms** | **REPORTED** |
| **P95 Generation Latency** | - | **12819.8 ms** | **REPORTED** |
| **Mean Generation Latency** | - | **4724.9 ms** | **REPORTED** |
| **Failover Rate** | Resilient | **0.0%** (0/135 calls) | **100% Primary Available** |

---

## 3. Breakdown by Question Type

| Question Type | Questions | Trials | Parse Success | Abstain Rate | Hallucinated Cites | P50 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **single-chunk** | 27 | 81 | 100.0% | 4.9% (4/81) | 0.0% (0) | 2890.2 |
| **multi-chunk** | 9 | 27 | 100.0% | 29.6% (8/27) | 0.0% (0) | 10240.5 |
| **unanswerable** | 9 | 27 | 100.0% | 100.0% (27/27) | 0.0% (0) | 1245.0 |

### Key Observations:
- **Unanswerable Questions (100% Precision):** Across all 9 unanswerable questions (27 trials), the pipeline reliably identified the absence of relevant facts and abstained with `citations: []` and an average response latency of ~1.2s.
- **Multi-Chunk Queries:** Multi-chunk synthesis queries require complex reasoning across diverse sections (e.g. connecting React frontend, Android NDK symbols, and Gradle plugins). These queries exhibit higher deliberative latency (P50 ~10.2s) and an elevated false abstain rate (29.6%), primarily where first-stage hybrid retrieval delivered incomplete candidate coverage in the top-5.

---

## 4. In-Depth Engineering Findings & Production Takeaways

### 4.1 The Extended Thinking Token Hazard
In modern reasoning models (Claude 3.7 Sonnet, Claude 3.5/Haiku with thinking), `max_tokens` represents the **combined ceiling** for reasoning tokens and visible output tokens:
$$\text{max\_tokens} = \text{thinking\_tokens} + \text{text\_tokens}$$

- **The Truncation Mechanism:** When reasoning effort is configured to `xhigh`, the model executes an internal reasoning trace that consumes 2000 to 3500 tokens on multi-chunk synthesis questions. Under an initial default of `max_tokens: 2048`, thinking tokens exhausted the entire allowance, leaving 0 tokens for visible output (`stop_reason: max_tokens`). On long queries, this produced partial strings severed mid-JSON (e.g. `{"answer": "Web (React...`), resulting in `Unterminated string` JSON parse errors (88.9% parse rate).
- **The Architectural Fix:** Bumping `max_tokens` to `8192` provides necessary headroom. Crucially, API billing in LLM providers is calculated strictly per **actual generated token**, not declared capacity. Short single-chunk and unanswerable queries terminate in 800-1200 total tokens (`stop_reason: end_turn`), incurring zero additional cost while completely eliminating JSON truncation (100.0% parse rate).

### 4.2 Reasoning Effort vs Latency Trade-Offs
- For single-shot grounded RAG, `effort: "xhigh"` ensures thorough verification against candidate chunks and 0.0% hallucinated citations.
- However, `xhigh` increases P95 latency to ~12.8s due to multi-thousand-token thinking chains. For latency-sensitive production serving, benchmarking `effort: "medium"` or `"high"` represents an actionable Pareto improvement to halve latency without compromising grounding.

### 4.3 Output Length Governance: System Prompt vs Token Budget
- **System Prompt Role:** Prompt rules such as *"Be direct, technical, and concise"* effectively constrain the visible `text_tokens`, minimizing latency and output token cost (which is 4-5x more expensive than input tokens).
- **Token Budget Role:** Prompt instructions cannot directly constrain internal thinking tokens in autonomous reasoning models. System prompt instructions and API token budgeting serve distinct, complementary purposes and must both be configured.

### 4.4 Failover Reliability
- Dual provider failover in `src/llm.py` provides resilient production gating. Configuring Claude Haiku as default with automated fallback to Gemini Flash resolves transient cloud availability spikes (e.g. Gemini 503 UNAVAILABLE errors observed during earlier sweeps) while maintaining identical Pydantic output schemas.

---

## 5. Addendum: Reasoning Effort Ablation (xhigh vs high vs medium)

To determine the optimal latency vs quality operating point on the Pareto frontier, we evaluated the single-shot generation pipeline across three reasoning effort configurations (`xhigh`, `high`, and `medium`) over all 45 dev questions (3 trials each = 135 trials per sweep, 405 total runs).

### 5.1 Comparative Metrics Table

| Effort Level | Parse Success Rate | Hallucinated Citations Rate | Abstain Accuracy (Unanswerable) | False Abstain Rate | P50 Latency (ms) | P95 Latency (ms) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **xhigh** | 100.0% (135/135) | 0.0% (0/204) | 100.0% (27/27) | 11.1% (12/108) | 3170.1 | 12819.8 | 4724.9 |
| **high** | 100.0% (135/135) | 0.0% (0/196) | 100.0% (27/27) | 12.0% (13/108) | 2370.8 | 7613.3 | 2968.6 |
| **medium** | 100.0% (135/135) | 0.0% (0/181) | 100.0% (27/27) | 13.9% (15/108) | 1682.6 | 5211.1 | 2333.4 |

### 5.2 Technical Takeaways

1. **Latency Delta and Tail Truncation:**
   - **`high` vs `xhigh`:** P50 latency drops by 25.2% (2370.8 ms vs 3170.1 ms), while P95 tail latency drops by 40.6% (7613.3 ms vs 12819.8 ms). Mean latency improves by 37.2% (2968.6 ms vs 4724.9 ms).
   - **`medium` vs `xhigh`:** P50 latency drops by 46.9% (1682.6 ms vs 3170.1 ms), while P95 latency drops by 59.4% (5211.1 ms vs 12819.8 ms). Mean latency drops by 50.6% (2333.4 ms vs 4724.9 ms).
   - The primary driver of latency reduction is the compression of internal thinking tokens during complex multi-chunk reasoning. Single-chunk and unanswerable queries remain predictably fast across all effort settings (~1.0s to ~2.5s).

2. **Quality Retention:**
   - **Grounding and Hallucination Resistance:** Across all 405 runs across all three sweeps, the hallucinated citation rate remained strictly **0.0%** (zero invalid chunk IDs cited).
   - **Negative Abstain Precision:** Abstain accuracy on unanswerable questions remained **100.0%** (27/27) across all three settings. The model reliably identifies when requested facts are absent from the retrieved candidate context without relying on maximum deliberative reasoning depth.
   - **Answerable Recall:** False abstain rates increase slightly as reasoning capacity is constrained: 11.1% (`xhigh`) to 12.0% (`high`) to 13.9% (`medium`). This incremental delta occurs exclusively on multi-chunk synthesis questions where partial retrieval requires complex multi-step hypothesis verification.

3. **Production Serving Recommendations:**
   - **Interactive User-Facing QA (Recommended: `effort: high`):** Provides the optimal operational trade-off. Shaves over 5.2 seconds off P95 tail latency (7.6s vs 12.8s) and brings mean response time under 3.0s, with negligible impact on answerable recall (+0.9% false abstains) and identical zero-hallucination guarantees.
   - **High-Throughput / Cost-Sensitive Batch Pipelines (Alternative: `effort: medium`):** Recommended when strict sub-2.0s median response times are mandated or token throughput costs dominate, accepting a 2.8% increase in false abstains for complex synthesis tasks.
   - **Complex Offline Synthesis / Research Agents (`effort: xhigh`):** Recommended for asynchronous agentic workflows or complex document summarization where latency SLAs are unconstrained and maximal multi-hop deduction across distant contexts is required.


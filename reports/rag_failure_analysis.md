# RAG Failure Analysis & Diagnostic Report

- **Ablation Timestamp:** `2026-11-05T13:47:22.300258+00:00`
- **Dataset:** `eval/golden_set.json` (`dev` split: `45` total, `36` answerable)
- **Ablation Records Source:** `runs/ablation_results.json`
- **Multi-Trial Generation Source:** `runs/transcripts/rag_dev_trials.jsonl` (`135` E2 trials)
- **Inspected Failure Cohort:** `22` failing/degraded records (`13` in production baseline `E3`, `9` in `E2`/`E1`/`E0`)

---

## 1. Executive Summary

End-to-end evaluation of the production RAG configuration (`E3`: Hybrid RRF + `BAAI/bge-reranker-base`) across `45` dev queries shows `13` degraded or failing records (correctness < 1.0, faithfulness < 1.0, or answerable recall@5 < 1.0 / false abstain). Combining these `13` E3 cases with `6` additional `E2` Hybrid RRF failures (verified across `135` single-shot trials) and `3` cross-ablation `E0`/`E1` diagnostic cases yields `22` evidence-audited failure records spanning all six taxonomy categories and all five stages of the diagnostic cascade.

### Cross-Configuration Failure Footprint (`dev` split, n=45)

| Configuration | Degraded / Failing Records | Correctness < 1.0 | Answerable Recall@5 < 1.0 | False Abstains | Faithfulness < 1.0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **E0** | 18 / 45 | 17 | 12 | 8 | 1 |
| **E1** | 16 / 45 | 15 | 11 | 6 | 0 |
| **E2** | 17 / 45 | 15 | 13 | 4 | 0 |
| **E3** | 13 / 45 | 11 | 11 | 2 | 0 |

- **E2 Multi-Trial Stability (`rag_dev_trials.jsonl`):** `12` / `108` answerable trials (`11.1%`) produced deterministic false abstains across `4` items: `gold-004`, `gold-015`, `gold-022`, `gold-037` (all 4 abstained in 3/3 trials).
- **Corpus Ingestion Audit (`data/bm25_index.pkl`, n=14,148 chunks):** `4,895` chunks (`34.6%`) lack document-title topic keywords in `chunk.text` (stored only in `heading_path`), `120` chunks (`0.8%`) contain unstripped JS `export` statements, and `150` chunks (`1.1%`) contain residual HTML layout tags (`<div>`, `<span>`).

---

## 2. Failure Category Counts (6-Category Taxonomy)

| Failure Category | E3 Baseline (n=13) | E3 Share | Full Inspected Cohort (n=22) | Cohort Share | Primary Diagnostic Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `retrieval miss` | 6 | 46.2% | 7 | 31.8% | Target chunk absent from Stage-1 top-50 candidate pool. |
| `ranking miss` | 5 | 38.5% | 8 | 36.4% | Target chunk present in Stage-1/other-mode candidates but ranked outside top-5. |
| `chunking cut` | 1 | 7.7% | 2 | 9.1% | Answer severed across chunk boundaries or broken by MDX/HTML truncation. |
| `model ignored context` | 1 | 7.7% | 3 | 13.6% | Target chunk present in top-5 prompt context, yet LLM omitted key facts. |
| `judge error` | 0 | 0.0% | 1 | 4.5% | LLM judge over-penalized a complete answer on unasked reference details. |
| `wrong golden label` | 0 | 0.0% | 1 | 4.5% | Gold URL set omitted an equivalent canonical documentation page. |
| **Total** | **13** | **100.0%** | **22** | **100.0%** | - |

---

## 3. Five-Level Diagnostic Cascade (`Ingestion -> Retrieval -> Rerank -> Attention -> Generation`)

| Cascade Level | E3 Baseline (n=13) | E3 Share | Full Inspected Cohort (n=22) | Cohort Share | Observed Failure Signature |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ingestion** | 1 | 7.7% | 3 | 13.6% | MDX `export`/HTML pollution, `<QuestLogItem>` splits, `###` fragmentation, narrow gold URLs. |
| **Retrieval** | 6 | 46.2% | 6 | 27.3% | Single-query embeddings/BM25 saturated top-50 with dominant topics or missed synonyms. |
| **Rerank** | 5 | 38.5% | 8 | 36.4% | Cross-encoder (`bge-reranker-base`) or RRF pushed table/heading-dependent chunks below rank #5. |
| **Attention** | 1 | 7.7% | 3 | 13.6% | Generator dropped 1-2 details or list bullets from a rank-#1 chunk inside `<retrieved_context>`. |
| **Generation** | 0 | 0.0% | 2 | 9.1% | Judge over-penalization on secondary details (`E1`) or ungrounded extrapolation (`E0`). |
| **Total** | **13** | **100.0%** | **22** | **100.0%** | - |

---

## 4. Detailed Case-by-Case Diagnostic Analysis

### 4.1 Production Baseline (`E3`) Degraded & Failing Cases (n=13)

#### `[E3] gold-003` (single-chunk) - `model ignored context` | Cascade: `Attention`
- **Question:** PostHog AI keeps giving me the wrong results when I ask about signups. What should I change in how I phrase my requests?
- **Metrics:** Hit@5=`1.00` | Recall@5=`1.00` | MRR=`1.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/product-analytics/analyze-data-ai`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/product-analytics/analyze-data-ai`, `https://posthog.com/docs/ai-observability/traces`, `https://posthog.com/docs/alerts`, `https://posthog.com/docs/slack`, `https://posthog.com/docs/replay-vision/troubleshooting`
- **Retrieved Chunk IDs:** `product-analytics/analyze-data-ai.mdx#4`, `ai-observability/traces.mdx#7`, `alerts/index.mdx#5`, `slack/index.mdx#1`, `replay-vision/troubleshooting.mdx#17`
- **Judge Reasoning (Correctness):** The reference lists five pieces of guidance: name events explicitly, specify time ranges, describe the visualization, iterate with follow-ups, and ask questions rather than just giving commands. The candidate covers the first four accurately and with matching examples. It omits the fifth, asking questions like 'Why did signups drop?'...
- **Stage-1 & Retrieval Evidence:** Target chunk product-analytics/analyze-data-ai.mdx#4 ranked #1 in Dense, #2 in BM25, #1 in Hybrid RRF, and #1 in E3 Cross-Encoder reranking (score 0.4685).
- **Chunk & Doc Evidence:** Chunk #4 explicitly lists all 5 tips as bullet points, ending with 'Ask questions, not just commands'. E2 included all 5 tips (correctness 1.0), whereas E3 emitted only the first 4 bullets and dropped the 5th.
- **Verified Root Cause:** Target chunk was at rank #1 in the prompt context, but single-shot generation omitted the final bullet point from the list.

#### `[E3] gold-004` (single-chunk) - `ranking miss` | Cascade: `Rerank`
- **Question:** I need to clean up a lot of old dashboards through the MCP. Is there a single call that removes several at once, and is there a limit?
- **Metrics:** Hit@5=`0.00` | Recall@5=`0.00` | MRR=`0.00` | Correctness=`0.0` | Faithfulness=`1.0` | Abstain=`True`
- **Gold URLs:** `https://posthog.com/docs/product-analytics/surfaces/mcp`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/vscode-extension/stale-flag-cleanup`, `https://posthog.com/docs/product-analytics/dashboards`, `https://posthog.com/docs/product-analytics/dashboards`, `https://posthog.com/docs/model-context-protocol/faq`, `https://posthog.com/docs/mcp-analytics/installation/other-languages`
- **Retrieved Chunk IDs:** `vscode-extension/stale-flag-cleanup.mdx#10`, `product-analytics/dashboards.mdx#8`, `product-analytics/dashboards.mdx#16`, `model-context-protocol/faq.mdx#18`, `mcp-analytics/installation/other-languages.mdx#4`
- **Judge Reasoning (Correctness):** False abstain on answerable question.
- **Stage-1 & Retrieval Evidence:** Target chunk product-analytics/surfaces/mcp.mdx#5 ranked #3 in Dense (E0 correctness 1.0), 0 matches in BM25 top-50, #8 in Stage-1 Hybrid RRF, and #6 in E3 Cross-Encoder (score 0.0084) - one slot outside the top-5 cutoff.
- **Chunk & Doc Evidence:** Chunk #5 ('## Dashboard tools') contains the dashboards-bulk-delete table row but does not contain the token 'MCP' anywhere in body text; 'MCP' exists only in heading_path ('Use product analytics over PostHog MCP > Dashboard tools'), which neither BM25 nor CrossEncoder inspects.
- **Verified Root Cause:** Omission of heading_path from BM25 and CrossEncoder inputs caused an exact-match dense hit (#3) to be diluted in RRF (#8) and ranked #6 just outside top-5.

#### `[E3] gold-007` (single-chunk) - `retrieval miss` | Cascade: `Retrieval`
- **Question:** My feature flag returned a value I didn't expect for one of my users. Is the flag misbehaving, or should I be looking somewhere else?
- **Metrics:** Hit@5=`0.00` | Recall@5=`0.00` | MRR=`0.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/feature-flags/best-practices`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/feature-flags/troubleshooting`, `https://posthog.com/docs/api/feature-flags/feature_flags_retrieve`, `https://posthog.com/docs/feature-flags/surfaces/web-app`, `https://posthog.com/docs/feature-flags/troubleshooting`, `https://posthog.com/docs/feature-flags/surfaces/api`
- **Retrieved Chunk IDs:** `feature-flags/troubleshooting.mdx#11`, `api/feature-flags/feature_flags_retrieve.mdx#0`, `feature-flags/surfaces/web-app.mdx#3`, `feature-flags/troubleshooting.mdx#1`, `feature-flags/surfaces/api.mdx#2`
- **Judge Reasoning (Correctness):** The reference says the flag itself is usually fine because evaluation is deterministic (same flag + same distinct ID gives the same result), and that unexpected values almost always come from the inputs: identity, properties, or flag definition. The candidate begins by saying the docs don't say whether the flag misbehaves, so it never...
- **Stage-1 & Retrieval Evidence:** Target chunk feature-flags/best-practices.mdx#5 did not enter Stage-1 top-50 in Dense, BM25, or Hybrid RRF (only sibling chunks #1, #0, #17, #14 appeared at ranks 14-42).
- **Chunk & Doc Evidence:** Section #5 explains deterministic hashing and input root causes using conversational wording, whereas feature-flags/troubleshooting.mdx#1/#11 directly match query keywords ('feature flag', 'unexpected', 'user') and crowded out best-practices.mdx#5.
- **Verified Root Cause:** Stage-1 dense and lexical retrievers saturated top-50 with troubleshooting chunks, failing to retrieve the deterministic hashing guidance chunk.

#### `[E3] gold-014` (single-chunk) - `retrieval miss` | Cascade: `Retrieval`
- **Question:** A self-driving report flagged a pattern in my product. What can I do with it, and does PostHog change anything automatically?
- **Metrics:** Hit@5=`0.00` | Recall@5=`0.00` | MRR=`0.00` | Correctness=`0.0` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/session-replay/surfaces/web-app`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/self-driving`, `https://posthog.com/docs/ai-engineering/ai-wizard`, `https://posthog.com/docs/health-checks/pageleave-events`, `https://posthog.com/docs/health-checks/no-live-events`, `https://posthog.com/docs/self-driving/inbox`
- **Retrieved Chunk IDs:** `self-driving/index.mdx#0`, `ai-engineering/ai-wizard.mdx#3`, `health-checks/pageleave-events.mdx#4`, `health-checks/no-live-events.mdx#4`, `self-driving/inbox/index.mdx#3`
- **Judge Reasoning (Correctness):** The reference says you can read the report (pattern, affected users/sessions, start time, replays), watch example recordings, discuss with an agent, follow the activity log, then fix config, redesign the step, hand it to an agent, or dismiss. It also says nothing happens until you decide, so nothing is changed automatically. The...
- **Stage-1 & Retrieval Evidence:** Target chunk session-replay/surfaces/web-app.mdx#6 scored 0 matches in Dense top-50, #43 in BM25 top-50, and missed Stage-1 Hybrid top-50 entirely (only #3 appeared at #38).
- **Chunk & Doc Evidence:** Chunk #6 ('### What you can do with a report') lacks the terms 'self-driving' and 'automatically' in body text ('Self-driving reports' is only in heading_path, while 'Why there is no pull request' is split into #5). Generic self-driving/index.mdx#0 dominated retrieval and caused an opposite answer ('opens a pull request').
- **Verified Root Cause:** Missing heading_path context and terminology overlap with product-wide self-driving docs prevented the session-replay report chunk from entering Stage-1 Hybrid top-50.

#### `[E3] gold-021` (single-chunk) - `ranking miss` | Cascade: `Rerank`
- **Question:** Which Cisco Meraki tables sync incrementally, and which ones do a full refresh on every sync?
- **Metrics:** Hit@5=`1.00` | Recall@5=`1.00` | MRR=`1.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/data-pipelines/sources/cisco-meraki`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/data-pipelines/sources/cisco-meraki`, `https://posthog.com/docs/data-pipelines/sources/instagram`, `https://posthog.com/docs/data-pipelines/sources/commercetools`, `https://posthog.com/docs/data-pipelines/sources/onesignal`, `https://posthog.com/docs/data-pipelines/sources/splunk-observability-cloud`
- **Retrieved Chunk IDs:** `data-pipelines/sources/cisco-meraki.md#0`, `data-pipelines/sources/instagram.md#3`, `data-pipelines/sources/commercetools.md#5`, `data-pipelines/sources/onesignal.md#3`, `data-pipelines/sources/splunk-observability-cloud.md#3`
- **Judge Reasoning (Correctness):** The question asks which tables are incremental and which are full refresh. The candidate correctly identifies network_events as the only incremental table, which matches the reference. It says the documentation does not name the full-refresh tables and declines to list them, whereas the reference lists six: networks, devices,...
- **Stage-1 & Retrieval Evidence:** Target chunk data-pipelines/sources/cisco-meraki.md#4 ranked #4 in Dense, #4 in BM25, and #1 in Stage-1 Hybrid RRF (E0/E1/E2 all achieved correctness 1.0), but E3 Cross-Encoder demoted #4 to rank #10 (score 0.9928) behind #0 (0.9992) and 4 other connector sync docs.
- **Chunk & Doc Evidence:** Chunk #4 is a Markdown table listing all 7 Meraki tables and sync methods, while #0 is prose overview lacking the 6 full-refresh table names. Because URL-level recall@5 matched #0, recall@5 stayed 1.0 despite losing the table chunk.
- **Verified Root Cause:** Cross-encoder reranker preferred prose sync-mode descriptions from unrelated connectors over the structured Markdown table in #4, demoting #4 from #1 to #10.

#### `[E3] gold-022` (single-chunk) - `ranking miss` | Cascade: `Rerank`
- **Question:** I'm connecting Mintlify as a data source in PostHog. Which of its tables sync incrementally, and what field do they use to track new or updated records?
- **Metrics:** Hit@5=`0.00` | Recall@5=`0.00` | MRR=`0.00` | Correctness=`0.0` | Faithfulness=`1.0` | Abstain=`True`
- **Gold URLs:** `https://posthog.com/docs/data-pipelines/sources/mintlify`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/data-pipelines/sources/crunchbase`, `https://posthog.com/docs/data-pipelines/sources`, `https://posthog.com/docs/data-pipelines/sources/azure-application-insights`, `https://posthog.com/docs/data-pipelines/sources/lightspeed-retail`, `https://posthog.com/docs/data-pipelines/sources/shipstation`
- **Retrieved Chunk IDs:** `data-pipelines/sources/crunchbase.md#4`, `data-pipelines/sources/index.mdx#7`, `data-pipelines/sources/azure-application-insights.md#3`, `data-pipelines/sources/lightspeed-retail.md#4`, `data-pipelines/sources/shipstation.md#2`
- **Judge Reasoning (Correctness):** False abstain on answerable question.
- **Stage-1 & Retrieval Evidence:** Target chunk data-pipelines/sources/mintlify.md#3 ranked #9 in BM25 and #20 in Stage-1 Hybrid RRF (while #2 ranked #1 in Dense and #1 in Hybrid). E3 Cross-Encoder demoted #2 to rank #9 (0.9682) and #3 to rank #18 (0.9096), yielding 0 Mintlify chunks in top-5.
- **Chunk & Doc Evidence:** Chunk #3 ('## Available tables') never mentions the word 'Mintlify' in body text (it resides solely in heading_path 'Linking Mintlify as a source > Available tables'). Reranker instead promoted crunchbase.md#4, index.mdx#7, lightspeed-retail.md#4, and shipstation.md#2.
- **Verified Root Cause:** Without heading_path prepended to candidate text, the cross-encoder could not bind the Mintlify table chunk to the entity 'Mintlify' and pushed both Mintlify chunks out of top-5.

#### `[E3] gold-037` (multi-chunk) - `ranking miss` | Cascade: `Rerank`
- **Question:** Our team is shipping a risky new checkout flow. We want to first expose it to a small subset of users, watch error rates and usage metrics, and widen the rollout only once we're satisfied. We also want beta testers to opt themselves in and out. Can we use early access feature management for both goals if our rollout targets company-level Groups and the feature is gated in our Python backend?
- **Metrics:** Hit@5=`1.00` | Recall@5=`0.50` | MRR=`1.00` | Correctness=`1.0` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/feature-flags/early-access-feature-management`, `https://posthog.com/docs/feature-flags/canary-release`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/feature-flags/early-access-feature-management`, `https://posthog.com/docs/libraries/js/usage`, `https://posthog.com/docs/feature-flags/creating-feature-flags`, `https://posthog.com/docs/feature-flags/user-and-group-targeting`, `https://posthog.com/docs/feature-flags`
- **Retrieved Chunk IDs:** `feature-flags/early-access-feature-management.mdx#0`, `libraries/js/usage.mdx#16`, `feature-flags/creating-feature-flags.mdx#10`, `feature-flags/user-and-group-targeting.mdx#15`, `feature-flags/index.mdx#0`
- **Judge Reasoning (Correctness):** The reference says early access is only available in the JS Web SDK and doesn't support Groups, so it can't cover group targeting or a Python backend gate. The canary release goal needs a different rollout approach. The beta opt-in matches early access but only for individual users in the JS Web SDK. The candidate gives the same...
- **Stage-1 & Retrieval Evidence:** First gold URL early-access-feature-management.mdx#0 ranked #1 in E3, while second gold URL feature-flags/canary-release.md#4 ranked #1 in BM25, #10 in Stage-1 Hybrid RRF, and #6 in E3 Cross-Encoder (score 0.0266) - one slot outside top-5 (recall@5=0.5, correctness=1.0).
- **Chunk & Doc Evidence:** Cross-encoder filled slots #2-#5 with JS usage and general feature-flags chunks (which incidentally supplied staged-rollout guidance for correctness=1.0), leaving canary-release.md#4 at #6 and #1 at #12.
- **Verified Root Cause:** Secondary gold document canary-release.md was present in the Stage-1 pool (#10) but stopped at rank #6 under the k=5 rerank cutoff.

#### `[E3] gold-038` (multi-chunk) - `retrieval miss` | Cascade: `Retrieval`
- **Question:** Our team is about to launch a checkout experiment. Before we go live, I want to confirm that my colleagues and I actually see the test variant when we visit the site, including when we're logged out. I also need to know how long the experiment has to run to get reliable results. What steps should I take for each of these, and what inputs determine the running time?
- **Metrics:** Hit@5=`0.00` | Recall@5=`0.00` | MRR=`0.00` | Correctness=`0.0` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/experiments/testing-and-launching`, `https://posthog.com/docs/experiments/sample-size-running-time`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/experiments/start-here`, `https://posthog.com/docs/experiments/no-code-web-experiments`, `https://posthog.com/docs/replay-vision/scanner-types`, `https://posthog.com/docs/experiments/no-code-web-experiments`, `https://posthog.com/docs/experiments/surfaces/mcp`
- **Retrieved Chunk IDs:** `experiments/start-here.mdx#0`, `experiments/no-code-web-experiments.md#7`, `replay-vision/scanner-types.mdx#8`, `experiments/no-code-web-experiments.md#8`, `experiments/surfaces/mcp.mdx#2`
- **Judge Reasoning (Correctness):** The reference answer says to add an optional override to the experiment's feature flag with an email condition set at 100% rollout. For logged-out users it uses a utm_source parameter in the URL, sets the override to the target variant, and saves. For running time it says to use the running time calculator, which depends on traffic and...
- **Stage-1 & Retrieval Evidence:** Neither experiments/testing-and-launching.mdx#0 nor experiments/sample-size-running-time.mdx#0 entered Stage-1 Hybrid top-50 (Dense had only sample-size-running-time.mdx#4 at #43; BM25 and Hybrid had 0 gold URL matches in top-50).
- **Chunk & Doc Evidence:** Both #0 chunks begin with unstripped JS 'export const ...' Cloudinary URL strings (6 lines in sample-size-running-time.mdx#0, ~84% token noise) because clean_mdx() requires trailing semicolons; testing-and-launching.mdx#0 also omits 'experiment' from its body header.
- **Verified Root Cause:** Ingestion export-statement noise and vocabulary mismatch against no-code preview docs kept both gold documents out of Stage-1 Hybrid top-50.

#### `[E3] gold-039` (multi-chunk) - `chunking cut` | Cascade: `Ingestion`
- **Question:** Our app renders a WebGL-based data visualization that we want to see in session replays so we can debug user issues. Before we turn that on, what do we need to verify about our PostHog JS SDK version and our PII exposure, and how should we think about keeping our Session Replay bill under control?
- **Metrics:** Hit@5=`0.00` | Recall@5=`0.00` | MRR=`0.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/session-replay/start-here`, `https://posthog.com/docs/session-replay/canvas-recording`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/session-replay`, `https://posthog.com/docs/logs/installation/javascript`, `https://posthog.com/docs/vscode-extension/session-replay`, `https://posthog.com/docs/session-replay`, `https://posthog.com/docs/session-replay/troubleshooting`
- **Retrieved Chunk IDs:** `session-replay/index.mdx#2`, `logs/installation/javascript.mdx#5`, `vscode-extension/session-replay.mdx#2`, `session-replay/index.mdx#1`, `session-replay/troubleshooting.mdx#2`
- **Judge Reasoning (Correctness):** The reference requires: (1) SDK v1.105.7+ for canvas capture, (2) canvas not covered by DOM masking, so confirm no PII or set up canvas masking, enable canvas recording in settings, 4 fps, (3) cost: free allowance, per-recording billing with volume discounts, billing limits, recording rules and retention as levers. The candidate states...
- **Stage-1 & Retrieval Evidence:** Gold chunk session-replay/start-here.mdx#2 never entered Stage-1 top-50 in Dense, BM25, or Hybrid; start-here.mdx#0 ranked #7 in E3 Reranker and canvas-recording.mdx#0 ranked #29.
- **Chunk & Doc Evidence:** session-replay/start-here.mdx uses <QuestLogItem> MDX components without Markdown ## headings. clean_mdx() stripped <QuestLogItem title=...>, left lowercase <div>/<span> tags, and split_long_section() severed #2 with orphaned '</span></div>' tags and no pricing heading.
- **Verified Root Cause:** JSX component stripping and arbitrary token splitting severed the pricing/billing section into fragmented HTML-polluted #2, preventing retrieval alongside canvas-recording#0.

#### `[E3] gold-040` (multi-chunk) - `retrieval miss` | Cascade: `Retrieval`
- **Question:** Our team ships an Android app and wants a User Paths report showing how users move between screens. Will navigation data be captured by default without custom instrumentation, and how should we read the bars and numbers in the resulting paths chart?
- **Metrics:** Hit@5=`1.00` | Recall@5=`0.50` | MRR=`1.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/product-analytics/autocapture`, `https://posthog.com/docs/product-analytics/paths`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/product-analytics/paths`, `https://posthog.com/docs/product-analytics/insights`, `https://posthog.com/docs/libraries/react-native`, `https://posthog.com/docs/product-analytics/paths`, `https://posthog.com/docs/product-analytics/path-cleaning`
- **Retrieved Chunk IDs:** `product-analytics/paths.mdx#0`, `product-analytics/insights.mdx#6`, `libraries/react-native/index.mdx#7`, `product-analytics/paths.mdx#2`, `product-analytics/path-cleaning.mdx#0`
- **Judge Reasoning (Correctness):** The question has two parts: (1) whether Android navigation is captured by default without custom instrumentation, and (2) how to read the bars and numbers in the paths chart. The reference answers (1) yes, navigation, lifecycle and deep-link autocapture are on by default on Android, while interaction autocapture is unsupported. It...
- **Stage-1 & Retrieval Evidence:** Gold URL product-analytics/paths.mdx#0/#2 ranked #1 and #4 in E3, while second gold URL product-analytics/autocapture.mdx#3 missed Dense, BM25, and Hybrid top-50 completely (Dense had unrelated autocapture.mdx#18 at #38).
- **Chunk & Doc Evidence:** The multi-part query about Android navigation autocapture and User Paths saturated Stage-1 top-50 with paths.mdx and react-native/index.mdx chunks, starving the short Android bullet inside autocapture.mdx#3 ('## Supported SDKs').
- **Verified Root Cause:** Single-pass query retrieval over-indexed on User Paths and React Native screens, failing to surface the Android autocapture defaults chunk into Stage-1 top-50.

#### `[E3] gold-041` (multi-chunk) - `ranking miss` | Cascade: `Rerank`
- **Question:** Our paid campaigns tag traffic with UTM parameters, and I want to check whether visitors from one specific UTM source are converting after we set up a conversion goal. Where do I go in PostHog to look at this, which metrics will I see for that traffic compared to the previous period, and how can I investigate the sessions behind those numbers?
- **Metrics:** Hit@5=`1.00` | Recall@5=`0.50` | MRR=`1.00` | Correctness=`1.0` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/web-analytics/dashboard`, `https://posthog.com/docs/web-analytics/surfaces/web-app`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/web-analytics/surfaces/web-app`, `https://posthog.com/docs/web-analytics/start-here`, `https://posthog.com/docs/web-analytics/start-here`, `https://posthog.com/docs/web-analytics`, `https://posthog.com/docs/data/utm-segmentation`
- **Retrieved Chunk IDs:** `web-analytics/surfaces/web-app.mdx#0`, `web-analytics/start-here.mdx#1`, `web-analytics/start-here.mdx#0`, `web-analytics/index.mdx#1`, `data/utm-segmentation.mdx#2`
- **Judge Reasoning (Correctness):** The candidate matches the reference on all core points. It names Web analytics at app.posthog.com/web. It lists visitors, pageviews, sessions, average session duration and bounce rate, each compared to the previous period. It says to click the UTM source row to filter the whole dashboard. It says conversions and conversion rate appear...
- **Stage-1 & Retrieval Evidence:** First gold URL web-analytics/surfaces/web-app.mdx#0 ranked #1 in E3, while second gold URL web-analytics/dashboard.mdx#0 ranked #13 in BM25, #23 in Hybrid RRF, and #16 in E3 Cross-Encoder (score 0.5758), leaving recall@5=0.5 (correctness=1.0 via surfaces/web-app#0).
- **Chunk & Doc Evidence:** Reranker promoted noisy JSX card markup in web-analytics/start-here.mdx#1 (rank #2) and start-here.mdx#0 (rank #3) above dashboard.mdx#0.
- **Verified Root Cause:** Cross-encoder ranked generic web-analytics overview and JSX card chunks ahead of dashboard.mdx#0, keeping URL recall@5 at 0.5.

#### `[E3] gold-043` (multi-chunk) - `retrieval miss` | Cascade: `Retrieval`
- **Question:** Our product has a React web frontend built with Vite and a separate Android app that includes NDK native code. We want readable stack traces in PostHog for both. What do we need to configure for each platform, and what should our Android release process account for so that crashes stay symbolicated?
- **Metrics:** Hit@5=`1.00` | Recall@5=`0.50` | MRR=`0.20` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/error-tracking/upload-source-maps/react`, `https://posthog.com/docs/error-tracking/upload-mappings/android`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/error-tracking/upload-source-maps/react-native`, `https://posthog.com/docs/libraries/react-native`, `https://posthog.com/docs/error-tracking/upload-source-maps/flutter`, `https://posthog.com/docs/error-tracking`, `https://posthog.com/docs/error-tracking/upload-mappings/android`
- **Retrieved Chunk IDs:** `error-tracking/upload-source-maps/react-native.mdx#8`, `libraries/react-native/index.mdx#32`, `error-tracking/upload-source-maps/flutter.mdx#2`, `error-tracking/index.mdx#2`, `error-tracking/upload-mappings/android.mdx#1`
- **Judge Reasoning (Correctness):** The question has two parts: web (React + Vite) configuration and Android configuration plus release process. The candidate explicitly says it lacks documentation for the Vite web part and omits the answer (build.sourcemap: true, chunkId comment verification). The Android part and release process are covered accurately and match the...
- **Stage-1 & Retrieval Evidence:** Second gold URL error-tracking/upload-mappings/android.mdx#1 ranked #5 in E3, while first gold URL error-tracking/upload-source-maps/react.mdx#0 never entered Dense, BM25, or Hybrid top-50.
- **Chunk & Doc Evidence:** Dual-platform query ('React web frontend built with Vite' + 'Android app with NDK') pulled react-native.mdx (#1, #2) and android/flutter source-map chunks into top-50; react.mdx also lost 6 <_snippets> imports stripped by clean_mdx().
- **Verified Root Cause:** Without multi-part query decomposition, React Native and Android tokens crowded out error-tracking/upload-source-maps/react.mdx#0 from Stage-1 top-50.

#### `[E3] gold-044` (multi-chunk) - `retrieval miss` | Cascade: `Retrieval`
- **Question:** We've synced our Stripe charges into PostHog and want revenue per user next to their product activity in one SQL query. The synced amount column comes through as a string, so my arithmetic on it doesn't work. How do I get this working, and where can I check the column types before writing the query?
- **Metrics:** Hit@5=`1.00` | Recall@5=`0.50` | MRR=`1.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/data-warehouse`, `https://posthog.com/docs/data-warehouse/sql/data-access`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/data-warehouse`, `https://posthog.com/docs/data-warehouse/query`, `https://posthog.com/docs/revenue-analytics/connect-to-customers`, `https://posthog.com/docs/revenue-analytics/payment-platforms/stripe`, `https://posthog.com/docs/data-warehouse/surfaces/mcp`
- **Retrieved Chunk IDs:** `data-warehouse/index.mdx#0`, `data-warehouse/query.mdx#0`, `revenue-analytics/connect-to-customers.mdx#10`, `revenue-analytics/payment-platforms/stripe.mdx#0`, `data-warehouse/surfaces/mcp.mdx#9`
- **Judge Reasoning (Correctness):** The question has two parts: how to fix the string amount for arithmetic, and where to check column types. The reference says to convert with toFloat(amount) (or toInt / JSONExtractInt), aggregate, and join to events, and to check the Database and Properties tabs in Data management, or the SQL tab for external sources. The candidate...
- **Stage-1 & Retrieval Evidence:** First gold URL data-warehouse/index.mdx#0 ranked #1 in E3, while second gold URL data-warehouse/sql/data-access.mdx#1 missed Dense, BM25, and Hybrid top-50 entirely.
- **Chunk & Doc Evidence:** Query phrases 'amount column comes through as a string' and 'arithmetic' failed to match data-access.mdx#1 wording ('cast a string property', 'type conversion functions, such as`toString`, `toFloat`'), while revenue-analytics Stripe docs filled Stage-1 slots.
- **Verified Root Cause:** Lexical and semantic vocabulary gap around SQL casting ('arithmetic' vs 'type conversion') left data-warehouse/sql/data-access.mdx#1 absent from Stage-1 top-50.

### 4.2 Hybrid RRF (`E2`) & Dev-Trial Failures (n=6)

#### `[E2] gold-002` (single-chunk) - `model ignored context` | Cascade: `Attention`
- **Question:** If I remove an insight from one of my dashboards, does it get deleted from PostHog entirely? I want to clean up without accidentally losing it.
- **Metrics:** Hit@5=`1.00` | Recall@5=`1.00` | MRR=`1.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False` | Dev Trials Abstain: `0/3`
- **Gold URLs:** `https://posthog.com/docs/product-analytics/dashboards`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/product-analytics/dashboards`, `https://posthog.com/docs/product-analytics/dashboards`, `https://posthog.com/docs/product-analytics/surfaces/web-app`, `https://posthog.com/docs/api/oauth`, `https://posthog.com/docs/privacy/data-storage`
- **Retrieved Chunk IDs:** `product-analytics/dashboards.mdx#20`, `product-analytics/dashboards.mdx#8`, `product-analytics/surfaces/web-app.mdx#0`, `api/oauth.mdx#19`, `privacy/data-storage.mdx#6`
- **Judge Reasoning (Correctness):** The candidate matches the core claims: removal only affects the current dashboard, the insight persists in PostHog and on other dashboards, and the toast offers 'Delete insight everywhere', which removes it everywhere. It also states the delete option requires editor access and that view-only users can only remove the insight from the...
- **Stage-1 & Retrieval Evidence:** Target chunk product-analytics/dashboards.mdx#20 was at rank #1 in E2 Hybrid RRF (hit@5=1.0, recall@5=1.0, MRR=1.0).
- **Chunk & Doc Evidence:** Chunk #20 explicitly states 'This opens a confirmation dialog that lists any other dashboards using the insight before you confirm. Deletion can be undone after confirming.' E2 omitted both sentences (correctness 0.5), whereas E3 retained the confirmation dialog (correctness 1.0).
- **Verified Root Cause:** Even with the exact gold chunk at rank #1, single-shot generation dropped two secondary safety details present at the bottom of the chunk.

#### `[E2] gold-009` (single-chunk) - `model ignored context` | Cascade: `Attention`
- **Question:** Before I launch an A/B test, what should I check about exposure events, and when should the exposure event fire?
- **Metrics:** Hit@5=`1.00` | Recall@5=`1.00` | MRR=`1.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False` | Dev Trials Abstain: `0/3`
- **Gold URLs:** `https://posthog.com/docs/experiments/exposures`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/experiments/exposures`, `https://posthog.com/docs/experiments/troubleshooting`, `https://posthog.com/docs/experiments/exposures`, `https://posthog.com/docs/experiments/running-experiments-without-feature-flags`, `https://posthog.com/docs/experiments/exposures`
- **Retrieved Chunk IDs:** `experiments/exposures.mdx#16`, `experiments/troubleshooting.mdx#9`, `experiments/exposures.mdx#6`, `experiments/running-experiments-without-feature-flags.mdx#3`, `experiments/exposures.mdx#1`
- **Judge Reasoning (Correctness):** The reference has four points: (1) confirm exposure events are sent correctly for all variants, (2) fire at the moment users encounter the change, not at page/app load, (3) monitor exposure balance across variants against configured percentages, (4) document custom exposure triggers and required properties. The candidate covers (1),...
- **Stage-1 & Retrieval Evidence:** Target chunk experiments/exposures.mdx#16 was at rank #1 in E2 Hybrid RRF (hit@5=1.0, recall@5=1.0, MRR=1.0).
- **Chunk & Doc Evidence:** Chunk #16 lists 5 numbered best practices; E2 included items 1, 2, 4, and 5 while skipping item 3 ('Monitor exposure balance'). E3 retrieved #14 (SRM detection) alongside #16 and scored 1.0.
- **Verified Root Cause:** Generator skipped the middle list item (#3 exposure balance) inside the rank-#1 chunk during synthesis.

#### `[E2] gold-012` (single-chunk) - `chunking cut` | Cascade: `Ingestion`
- **Question:** I only want to record a percentage of the sessions in a trigger group, and I want to ignore very short sessions. How can I do that?
- **Metrics:** Hit@5=`1.00` | Recall@5=`1.00` | MRR=`1.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False` | Dev Trials Abstain: `0/3`
- **Gold URLs:** `https://posthog.com/docs/session-replay/how-to-control-which-sessions-you-record`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/session-replay/how-to-control-which-sessions-you-record`, `https://posthog.com/docs/session-replay/how-to-control-which-sessions-you-record`, `https://posthog.com/docs/session-replay/how-to-control-which-sessions-you-record`, `https://posthog.com/docs/session-replay/how-to-control-which-sessions-you-record`, `https://posthog.com/docs/libraries/js/usage`
- **Retrieved Chunk IDs:** `session-replay/how-to-control-which-sessions-you-record.mdx#1`, `session-replay/how-to-control-which-sessions-you-record.mdx#10`, `session-replay/how-to-control-which-sessions-you-record.mdx#7`, `session-replay/how-to-control-which-sessions-you-record.mdx#3`, `libraries/js/usage.mdx#19`
- **Judge Reasoning (Correctness):** The question has two parts: sampling a percentage and ignoring short sessions. The candidate answers the sampling part correctly: per-group sample rate, applied to matching sessions, deterministic by session ID, and posthog-js 1.85.0+. It does not say how to configure the minimum duration. It only says it is a per-group setting and...
- **Stage-1 & Retrieval Evidence:** E2 retrieved session-replay/how-to-control-which-sessions-you-record.mdx#1, #10, #7, #3 in top-5 (URL recall@5=1.0) while #8 ('### Minimum duration') sat outside top-5 at Hybrid #14.
- **Chunk & Doc Evidence:** chunk_document() split '## Trigger groups' on every '###' subheading into 10 chunks, separating '### Sampling' (#3) from '### Minimum duration' (#8). golden_set.json also tagged #2 ('### Controls available', a 6-row link table lacking values) as single-chunk.
- **Verified Root Cause:** Aggressive ### subheading splitting separated co-configured trigger-group controls across distant chunks, causing E2 to answer sampling and fail minimum duration.

#### `[E2] gold-015` (single-chunk) - `ranking miss` | Cascade: `Rerank`
- **Question:** Which built-in collection would show me sessions with multiple rage clicks or exceptions, and what time window does it cover?
- **Metrics:** Hit@5=`0.00` | Recall@5=`0.00` | MRR=`0.00` | Correctness=`0.0` | Faithfulness=`1.0` | Abstain=`True` | Dev Trials Abstain: `3/3`
- **Gold URLs:** `https://posthog.com/docs/session-replay/how-to-watch-recordings`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/data-pipelines/sources/posthog`, `https://posthog.com/docs/product-analytics/autocapture`, `https://posthog.com/docs/session-replay/surfaces/mcp`, `https://posthog.com/docs/session-replay/surfaces/mcp`, `https://posthog.com/docs/session-replay/surfaces/web-app`
- **Retrieved Chunk IDs:** `data-pipelines/sources/posthog.mdx#13`, `product-analytics/autocapture.mdx#11`, `session-replay/surfaces/mcp.mdx#2`, `session-replay/surfaces/mcp.mdx#3`, `session-replay/surfaces/web-app.mdx#2`
- **Judge Reasoning (Correctness):** False abstain on answerable question.
- **Stage-1 & Retrieval Evidence:** Target chunk session-replay/how-to-watch-recordings.mdx#10 ranked #1 in BM25 (E1) and #14 in Stage-1 Hybrid RRF, missing E2 top-5 and causing 3/3 false abstains in rag_dev_trials.jsonl (recovered to rank #1 and correctness 1.0 in E3).
- **Chunk & Doc Evidence:** Generic 'rage clicks' and 'sessions' chunks from data-pipelines/sources/posthog.mdx#13, autocapture.mdx#11, and mcp.mdx#2/#3 diluted RRF top-5 above #10.
- **Verified Root Cause:** RRF dilution pushed the exact built-in collections table (#10) to rank #14, triggering consistent false abstains across E2 ablation and all 3 dev trials.

#### `[E2] gold-042` (multi-chunk) - `ranking miss` | Cascade: `Rerank`
- **Question:** Our onboarding survey has partial responses enabled. We want a Slack message for each response, including people who drop off partway through. We also want to know whether PostHog automatically turns the open-text comments into engineering work. What steps set up the Slack alerts, what default must we change to include partial submissions, and what happens to the open-text comments on the Self-driving side?
- **Metrics:** Hit@5=`1.00` | Recall@5=`0.50` | MRR=`0.25` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False` | Dev Trials Abstain: `0/3`
- **Gold URLs:** `https://posthog.com/docs/surveys/start-here`, `https://posthog.com/docs/surveys/destinations`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/surveys/creating-surveys`, `https://posthog.com/docs/support/surfaces/slack`, `https://posthog.com/docs/self-driving/scouts`, `https://posthog.com/docs/surveys/destinations`, `https://posthog.com/docs/self-driving/inbox`
- **Retrieved Chunk IDs:** `surveys/creating-surveys.mdx#21`, `support/surfaces/slack.mdx#5`, `self-driving/scouts.mdx#14`, `surveys/destinations.mdx#1`, `self-driving/inbox/index.mdx#3`
- **Judge Reasoning (Correctness):** The question has three parts: the Slack setup steps, the default to change, and what happens to open-text comments on the Self-driving side. The candidate covers the first two correctly, matching the reference: Notifications tab, Add notification, Slack, channel, template with quick insert, buttons, and toggling off 'Only notify for...
- **Stage-1 & Retrieval Evidence:** First gold chunk surveys/destinations.mdx#1 ranked #4 in E2, while second gold chunk surveys/start-here.mdx#1 ranked #13 in BM25 and #29 in Stage-1 Hybrid RRF (recovered to rank #4 and correctness 1.0 in E3).
- **Chunk & Doc Evidence:** start-here.mdx#1 contains the exact Self-driving survey scout answer; #2 was severed mid-<QuestLogItem>. In E2 top-5, unrelated Slack/scouts chunks occupied ranks #2-#3 and #5.
- **Verified Root Cause:** Stage-1 Hybrid RRF buried surveys/start-here.mdx#1 at rank #29 behind generic Slack and Self-driving chunks, leaving E2 with only half the answer.

#### `[E2] gold-045` (multi-chunk) - `ranking miss` | Cascade: `Rerank`
- **Question:** Our finance team uses Slash and our product team uses Postmark, and we want both pulled into PostHog's data pipeline. For Slash, what credentials do I need to enter if I'm using a user-scoped key instead of an organization key? And which Postmark schema should I select to investigate email delivery failures?
- **Metrics:** Hit@5=`1.00` | Recall@5=`0.50` | MRR=`1.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False` | Dev Trials Abstain: `0/3`
- **Gold URLs:** `https://posthog.com/docs/data-pipelines/sources/slash`, `https://posthog.com/docs/data-pipelines/sources/postmark`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/data-pipelines/sources/slash`, `https://posthog.com/docs/model-context-protocol/faq`, `https://posthog.com/docs/feature-flags/targeting-groups`, `https://posthog.com/docs/settings/organizations`, `https://posthog.com/docs/web-analytics/surfaces/api`
- **Retrieved Chunk IDs:** `data-pipelines/sources/slash.md#1`, `model-context-protocol/faq.mdx#6`, `feature-flags/targeting-groups.md#0`, `settings/organizations.mdx#7`, `web-analytics/surfaces/api.mdx#4`
- **Judge Reasoning (Correctness):** The question has two parts. Part 1 (Slash credentials with a user-scoped key): the candidate correctly says to enter the API key and the Legal entity ID. It omits the setup steps, but the core answer is correct. Part 2 (Postmark schema for delivery failures): the reference says to use the bounces schema, and the candidate says it...
- **Stage-1 & Retrieval Evidence:** First gold chunk data-pipelines/sources/slash.md#1 ranked #1 in E2, while second gold chunk data-pipelines/sources/postmark.md#1 ranked #2 in BM25 and #6 in Stage-1 Hybrid RRF - one slot outside top-5 (recovered to rank #2 and correctness 1.0 in E3).
- **Chunk & Doc Evidence:** Dense search missed postmark.md#1 completely (0 matches in Dense top-50), so RRF ranked it #6 behind generic API-key and group-targeting chunks despite BM25 rank #2.
- **Verified Root Cause:** Dense miss on Postmark schema terminology pulled BM25 rank-#2 postmark.md#1 down to Hybrid RRF rank #6, one slot outside the top-5 window.

### 4.3 Cross-Ablation (`E0` / `E1`) Diagnostic Edge Cases (n=3)

#### `[E1] gold-024` (single-chunk) - `judge error` | Cascade: `Generation`
- **Question:** My data quality check on the events table keeps timing out. What should I do first, and does an errored check count as a failing assertion?
- **Metrics:** Hit@5=`1.00` | Recall@5=`1.00` | MRR=`1.00` | Correctness=`0.5` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/data-warehouse/data-quality/troubleshooting`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/data-warehouse/data-quality/troubleshooting`, `https://posthog.com/docs/data-warehouse/data-quality/start-here`, `https://posthog.com/docs/data-warehouse/data-quality/troubleshooting`, `https://posthog.com/docs/data-warehouse/data-quality/mcp`, `https://posthog.com/docs/data-warehouse/data-quality/mcp`
- **Retrieved Chunk IDs:** `data-warehouse/data-quality/troubleshooting.mdx#1`, `data-warehouse/data-quality/start-here.mdx#4`, `data-warehouse/data-quality/troubleshooting.mdx#6`, `data-warehouse/data-quality/mcp.mdx#3`, `data-warehouse/data-quality/mcp.mdx#4`
- **Judge Reasoning (Correctness):** The candidate matches the main claims: open the error details first, the list of common causes including timeout, enabling 'Only check recent rows' with events filtered on timestamp, use the smallest window, and an errored check is not a failing assertion. It omits the persons/groups created_at detail and the materialization gate note...
- **Stage-1 & Retrieval Evidence:** Target chunk data-warehouse/data-quality/troubleshooting.mdx#6 ranked #3 in E1 BM25 (hit@5=1.0, recall@5=1.0).
- **Chunk & Doc Evidence:** Candidate answered both parts asked for the events table accurately ('open error details first', 'Only check recent rows filters on timestamp', 'errored check does not count as failing assertion'). CorrectnessJudge explicitly hedged ('though it could be 1.0') and docked to 0.5 for omitting persons/groups created_at and unasked materialization-gate details.
- **Verified Root Cause:** Calibrated CorrectnessJudge over-penalized a complete answer by enforcing secondary reference sentences (persons/groups and materialization gates) not asked in the query.

#### `[E0] gold-025` (single-chunk) - `wrong golden label` | Cascade: `Ingestion`
- **Question:** My materialized view in the data warehouse stopped refreshing and I found the bug in its SQL. Do I need to open a pull request in my repo to fix it?
- **Metrics:** Hit@5=`0.00` | Recall@5=`0.00` | MRR=`0.00` | Correctness=`1.0` | Faithfulness=`1.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/data-warehouse/surfaces/web-app`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/health-checks/materialized-view-failures`, `https://posthog.com/docs/data-pipelines/sources/postgres`, `https://posthog.com/docs/data-warehouse/views/materialize-troubleshooting`, `https://posthog.com/docs/health-checks/materialized-view-failures`, `https://posthog.com/docs/data-warehouse/data-quality/block-materialization`
- **Retrieved Chunk IDs:** `health-checks/materialized-view-failures.mdx#0`, `data-pipelines/sources/postgres.mdx#47`, `data-warehouse/views/materialize-troubleshooting.mdx#8`, `health-checks/materialized-view-failures.mdx#3`, `data-warehouse/data-quality/block-materialization.mdx#0`
- **Judge Reasoning (Correctness):** The reference says no PR is needed because the view definition lives in PostHog. The fix is to apply the corrected query in the PostHog web app, and the check clears once a materialization succeeds. The candidate also says no PR is needed. It says to fix the view's SQL in PostHog (Data modeling), re-run the materialization, and confirm...
- **Stage-1 & Retrieval Evidence:** E0 retrieved health-checks/materialized-view-failures.mdx#0 and #3 at ranks #1 and #4, yielding recall@5=0.0 against gold_urls yet correctness=1.0 and faithfulness=1.0.
- **Chunk & Doc Evidence:** golden_set.json lists only https://posthog.com/docs/data-warehouse/surfaces/web-app as gold_url, but health-checks/materialized-view-failures.mdx#3 documents the identical in-app SQL fix steps in Data modeling without opening a repository PR.
- **Verified Root Cause:** Overly narrow gold_urls list omitted an equivalent canonical documentation page (health-checks/materialized-view-failures), causing a false-zero retrieval score.

#### `[E0] gold-038` (multi-chunk) - `retrieval miss` | Cascade: `Generation`
- **Question:** Our team is about to launch a checkout experiment. Before we go live, I want to confirm that my colleagues and I actually see the test variant when we visit the site, including when we're logged out. I also need to know how long the experiment has to run to get reliable results. What steps should I take for each of these, and what inputs determine the running time?
- **Metrics:** Hit@5=`0.00` | Recall@5=`0.00` | MRR=`0.00` | Correctness=`0.5` | Faithfulness=`0.0` | Abstain=`False`
- **Gold URLs:** `https://posthog.com/docs/experiments/testing-and-launching`, `https://posthog.com/docs/experiments/sample-size-running-time`
- **Retrieved URLs (Top-5):** `https://posthog.com/docs/experiments/best-practices`, `https://posthog.com/docs/experiments/start-here`, `https://posthog.com/docs/experiments/best-practices`, `https://posthog.com/docs/experiments/surfaces/web-app`, `https://posthog.com/docs/integrations/lovable`
- **Retrieved Chunk IDs:** `experiments/best-practices.mdx#3`, `experiments/start-here.mdx#0`, `experiments/best-practices.mdx#4`, `experiments/surfaces/web-app.mdx#2`, `integrations/lovable.mdx#7`
- **Judge Reasoning (Correctness):** The question has two parts: (1) verifying the variant before launch, including for logged-out users, and (2) running time and its determining inputs. Part 1: The reference says to add an optional override on the feature flag, with a condition set of email = a specific address at 100% rollout. For logged-out users, it says to use a...
- **Judge Reasoning (Faithfulness):** Claim check against the chunks: 1. Variant overrides to verify implementation before exposing real users: supported by the web-app chunk ("verify your implementation with variant overrides before you expose real users"). 2. Small rollout (e.g., 5%) for a few days, and the checklist (logging, crashes, session replays, 50/50 ratio):...
- **Stage-1 & Retrieval Evidence:** E0 Dense failed to retrieve either testing-and-launching.mdx#0 or sample-size-running-time.mdx#0 in top-5 (hit@5=0.0, recall@5=0.0) and produced the only faithfulness_score=0.0 record in all 180 ablation evaluations.
- **Chunk & Doc Evidence:** Missing testing-and-launching.mdx#0, the generator extrapolated from surfaces/web-app.mdx#2 by instructing the user to confirm logged-out behavior 'using the variant override workflow' - a claim unsupported by retrieved chunks.
- **Verified Root Cause:** Stage-1 retrieval miss combined with over-extrapolation during generation produced an ungrounded recommendation (faithfulness 0.0) for logged-out variant overrides.

---

## 5. Prioritized Fix Hypotheses (Top 3)

### Hypothesis 1 (Priority 1 - Highest Impact): Context-Aware Chunk Serialization & MDX Ingestion Cleanup
- **Target Failures:** `gold-004`, `gold-014`, `gold-021`, `gold-022`, `gold-038`, `gold-039`, `gold-041` (`7/13` E3 degraded cases).
- **Root Mechanism:** `src/ingest.py` (`build_indexes`) embeds and BM25-tokenizes `chunk.text` without `heading_path` or `title`, leaving `34.6%` of all corpus chunks (`4,895/14,148`) without their document-title topic keywords in `chunk.text` (e.g., `'MCP'` absent from `mcp.mdx#5`; `'Mintlify'` absent from `mintlify.md#3`). Similarly, `src/retrieval.py` feeds only `cand.text` to `BAAI/bge-reranker-base`. Meanwhile, `clean_mdx()` leaves semicolon-less `export const` URL lines (`120` chunks) and `<QuestLogItem>` / HTML tags (`150` chunks) intact.
- **Proposed Fix:** (1) Prepend `[Document: {heading_path}]` to chunk text for BM25 indexing, dense embedding, and cross-encoder reranking (`(query, f"{cand.heading_path}\n{cand.text}")`). (2) Update `clean_mdx()` to strip semicolon-optional `export` declarations, convert `<QuestLogItem title=...>` / `<Step title=...>` titles into Markdown subheadings, and strip residual HTML layout tags (`<div>`, `<span>`).
- **Expected Impact:** Directly resolves `gold-004`, `gold-021`, and `gold-022` reranker demotions and removes noise blocking `gold-014`, `gold-038`, and `gold-039`, targeting `+0.08` to `+0.14` E3 Recall@5 and `+0.06` to `+0.11` E3 Correctness on the `dev` split.
- **How to Validate:** Rebuild indexes via `python -m src.ingest`, verify `pytest tests/test_regression_gate.py` passes with zero regression, and re-run `python -m eval.eval_ablation` on `dev` (keeping `holdout` frozen) to confirm gains on the 7 target queries.

### Hypothesis 2 (Priority 2 - High Impact on Multi-Chunk Queries): Multi-Part Query Decomposition & Per-URL Diversification
- **Target Failures:** `gold-007`, `gold-037`, `gold-038`, `gold-040`, `gold-043`, `gold-044` (`6/13` E3 degraded cases; multi-chunk Recall@5 is `0.5000` vs `0.8519` single-chunk).
- **Root Mechanism:** Multi-part queries combining two products or workflows (e.g., React+Vite and Android NDK in `gold-043`; Android autocapture and User Paths in `gold-040`; Stripe SQL joins and casting in `gold-044`) saturate single-query dense and BM25 top-50 pools with one dominant topic, starving the second required document out of Stage-1.
- **Proposed Fix:** Add lightweight query decomposition for compound questions (generating 2-3 focused sub-queries, running hybrid retrieval per sub-query, fusing via RRF with `candidate_pool_size=100`) and enforce a soft per-URL diversity cap (`max_chunks_per_url=2`) in the top-5 reranked output.
- **Expected Impact:** Elevates secondary-document Stage-1 coverage on multi-chunk queries (`gold-037`, `gold-038`, `gold-040`, `gold-043`, `gold-044`), projected to lift multi-chunk Recall@5 from `0.5000` toward `0.78-0.89` and overall E3 Correctness from `0.8333` above `0.90`.
- **How to Validate:** Benchmark single-query E3 vs decomposed E3 on `eval/golden_set.json` (`dev` split) using `eval/eval_ablation.py`, checking multi-chunk Recall@5, Correctness, and P95 end-to-end latency within SLA.

### Hypothesis 3 (Priority 3 - Targeted Completeness & Eval Hygiene): Subsection Merging, List-Completeness Prompting & Judge/Gold Calibration
- **Target Failures:** `gold-003` (E3), `gold-002` / `gold-009` / `gold-012` (E2), `gold-024` (E1 `judge error`), `gold-025` (E0 `wrong golden label`).
- **Root Mechanism:** Two distinct micro-defects remain when retrieval succeeds: (a) `chunk_document()` splits small `###` sibling subsections into isolated chunks (`gold-012` separated `Sampling` `#3` from `Minimum duration` `#8`), and single-shot generation occasionally drops 1 of 5 bullet points from rank-#1 chunks (`gold-003`, `gold-002`, `gold-009`); (b) `CorrectnessJudge` occasionally docks answers for omitting unasked reference details (`gold-024`), and `gold_urls` omits equivalent canonical docs (`gold-025`).
- **Proposed Fix:** (1) Merge adjacent small `###` subsections under their parent `##` heading up to `CHUNK_MAX_TOKENS=500` during ingestion. (2) Add a multi-item completeness check in `src/generation.py` `SYSTEM_PROMPT`. (3) Refine `CorrectnessJudge` instructions to anchor completeness strictly to user-asked clauses, and audit `gold_urls` for equivalent canonical pages (`health-checks/materialized-view-failures` on `gold-025`; `gold-012` multi-chunk tag).
- **Expected Impact:** Eliminates `Attention` partial-credit drops (`0.5 -> 1.0` on `gold-003`, `gold-002`, `gold-009`), keeps co-configured controls intact (`gold-012`), and removes false metric/judge penalties (`gold-024`, `gold-025`).
- **How to Validate:** Re-run `eval/calibrate_judge.py` against `eval/human_labels_30.json` to preserve kappa = 1.000, execute 3-trial generation (`eval/run_generation.py`) to confirm zero bullet-omission regressions, and verify `eval/validate_golden_set.py` passes.

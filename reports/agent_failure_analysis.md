# Agent Trajectory Failure Analysis & Diagnostic Report

- **Model:** `claude-haiku-5-5`
- **Tasks & Transcripts:** `eval/agent_tasks.json` (`12` tasks x `3` trials = `36` eval trials in `runs/transcripts/agent_eval`)
- **Standalone Loop Transcripts:** `runs/transcripts/agent_trials.jsonl` (`3` runs)
- **Inspected Failing & Flaky Cohort:** `14` runs (`7` current eval failures, `3` resolved historical grader-bug failures, `4` flaky/multi-step runs)

---

## 1. Executive Summary

Deterministic offline re-evaluation of all `36` saved agent trajectories under `eval/eval_agent.py` yields `29/36` passing trials (**Pass@1 = 80.6%**, **Pass@3 = 91.7%**, **Pass^3 = 66.7%**), with `7` failing trials (`task-03_t2, task-10_t1, task-11_t1, task-11_t3, task-12_t1, task-12_t2, task-12_t3`).

### Re-Evaluated Benchmark vs Historical `reports/agent_eval.md`

| Metric / Check | Historical `reports/agent_eval.md` | Current Re-Evaluation (`eval_agent.py`) | Delta / Verified Cause |
| :--- | :---: | :---: | :--- |
| **Pass@1** | 72.2% (26/36) | 80.6% (29/36) | +8.4 pp (`task-05` synonym fix) |
| **Pass@3** | 83.3% (10/12) | 91.7% (11/12) | +8.4 pp (`task-05` 0/3 -> 3/3) |
| **Pass^3** | 58.3% (7/12) | 66.7% (8/12) | +8.4 pp (`task-05` reliability) |
| **`semantic_correctness`** | 75.0% (27/36) | 83.3% (30/36) | +8.3 pp |
| **`task-05` Trial Score** | `0/3` | `3/3` | `agent_tasks.json` updated from exact strings to synonym lists |

- **Historical Report Baseline:** Earlier summary metrics and narrative in reports/agent_eval.md reflected an earlier task-set snapshot (describing task-11 as 'IBM z/OS mainframe' and task-12 as 'quantum telemetry SLA'), whereas current eval/agent_tasks.json defines task-11 as CAN bus automotive telematics and task-12 as COBOL SDK / IBM z/OS mainframe (alongside synonym-list expansion on task-05).

### Task Type & Grader Breakdown (Current Re-Evaluation)

| Task Type | Tasks | Trials | Pass@1 | Pass@3 | Pass^3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `single_hop` | 8 | 24 | 95.8% | 100.0% | 87.5% |
| `multi_hop` | 2 | 6 | 83.3% | 100.0% | 50.0% |
| `unanswerable` | 2 | 6 | 16.7% | 50.0% | 0.0% |

| Grader | Pass Rate | Failing Count |
| :--- | :---: | :---: |
| `constraint_search_called` | 100.0% | 0 / 36 |
| `constraint_multihop_depth` | 100.0% | 0 / 36 |
| `constraint_step_budget` | 100.0% | 0 / 36 |
| `constraint_no_duplicate_queries` | 100.0% | 0 / 36 |
| `constraint_no_fabricated_citations` | 83.3% | 6 / 36 |
| `abstain_accuracy` | 83.3% | 6 / 36 |
| `citation_accuracy` | 83.3% | 6 / 36 |
| `semantic_correctness` | 83.3% | 6 / 36 |

- **Trajectory & Loop-Guard Telemetry (n=36 eval trials):** `0` intercepted or normalized duplicate queries; `72` total `search_docs` calls (mean `2.00`/trial; step distribution `{2: 29, 3: 7}`, call distribution `{1: 7, 2: 22, 3: 7}`, budget `6`); mean `tokens_in` = `8,987.0` (`7,185.3` for 1-2 calls vs `16,451.0` for 3+ calls, max `20,044` in `task-08_t1`); in-loop latency sum `279.17s`.
- **Standalone Trials (`agent_trials.jsonl`, n=3):** `0` duplicate queries; max tool calls `4`, max `tokens_in` `18,231` (standalone trial `2c815ecfbe79` from `runs/transcripts/agent_trials.jsonl` across 4 calls).

---

## 2. Failure Category Counts (5-Category + Abstention Taxonomy)

Note: Under `eval/eval_agent.py`, the `5` unanswerable over-helpfulness trials also fail `constraint_no_fabricated_citations` (`Fabricated N citations on unanswerable task`). Below, they are separated into `over-helpfulness / failed abstain` so distinct mechanisms remain visible.

| Failure Category | Current Failures (n=7) | Current Share | Historical Failures (n=10) | Inspected Cohort (n=14) | Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `bad query` | 0 | 0.0% | 0 | 4 | Under-specified or keyword-overloaded query missed target doc or triggered extra search turns. |
| `stopped early` | 1 | 14.3% | 1 | 1 | Terminated below MAX_STEPS with false abstention instead of reformulating query. |
| `loop or duplicate queries` | 0 | 0.0% | 0 | 0 | Repeated identical search query intercepted by loop guard or redundant loop. |
| `fabricated citation` | 1 | 14.3% | 1 | 1 | Cited a chunk ID absent from all_retrieved_chunk_ids (plus unanswerable policy overlap). |
| `grader bug` | 0 | 0.0% | 3 | 3 | Deterministic substring matcher falsely penalized valid synonym phrasing. |
| `over-helpfulness / failed abstain` | 5 | 71.4% | 5 | 5 | Answered unanswerable question ('No' + workaround) with abstain=false and citations. |
| **Total** | **7** | **100.0%** | **10** | **14** | - |

---

## 3. ReAct Loop Failure Taxonomy

| ReAct Failure Mode | Current Failures (n=7) | Current Share | Historical Failures (n=10) | Inspected Cohort (n=14) | Diagnostic Signature |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `duplicate queries` | 0 | 0.0% | 0 | 0 | Identical query repeated within a single trajectory. |
| `premature stop` | 1 | 14.3% | 1 | 1 | Halted search loop early with unused step budget despite missing evidence. |
| `excessive tool calls` | 0 | 0.0% | 0 | 4 | Issued redundant or off-target 3rd/4th search calls inflating token context. |
| `context rot` | 1 | 14.3% | 1 | 1 | High multi-step token/chunk load degraded grounding (e.g., #0 cited as #3). |
| `failed abstention / over-helpfulness` | 5 | 71.4% | 5 | 5 | Proposed workaround with citations instead of abstaining on unsupported capabilities. |
| `grader bug` | 0 | 0.0% | 3 | 3 | Evaluation oracle mismatch caused false-negative grading on valid trajectory. |
| **Total** | **7** | **100.0%** | **10** | **14** | - |

---

## 4. Top 2 Root Causes

### Root Cause 1: Unanswerable Over-Helpfulness & Negative-Question Abstention Mismatch
- **Failure Share:** `5/7` current eval failures (`71.4%`), `5/14` inspected runs (`35.7%`)
- **Affected Runs:** `task-11_t1`, `task-11_t3`, `task-12_t1`, `task-12_t2`, `task-12_t3`
- **Verified Mechanism:** When asked negative-capability questions (task-11 CAN bus, task-12 COBOL/z/OS), the agent retrieves general ingestion or supported-SDK documentation, answers 'No' while proposing custom HTTP/bridge workarounds, and outputs abstain=false with citations. Because eval_agent.py requires abstain=true and citations=[] on unanswerable tasks, 5 of 6 unanswerable trials fail (83.3% failure rate, 0.0% Pass^3).

### Root Cause 2: Query Sensitivity & Multi-Step Context Growth (Trajectory Instability)
- **Failure Share:** `2/7` current eval failures (`28.6%`), `6/14` inspected runs (`42.9%`)
- **Affected Runs:** `task-03_t2`, `task-10_t1`, `task-06_t2`, `task-08_t1`, `task-10_t2`, standalone trial `2c815ecfbe79` (from `runs/transcripts/agent_trials.jsonl`)
- **Verified Mechanism:** Across answerable tasks, single-token query shifts and multi-step retrieval accumulation drive answerable failures (task-03_t2, task-10_t1) plus 4 flaky multi-call runs whose input tokens double to 15,185-20,044 (mean 16,451.0 for 3+ calls vs 7,185.3 for 1-2 calls). Separately, historical task-05 (3/10 past failures) exposed brittle substring grading.

---

## 5. Detailed Per-Run Breakdown (n=14 Inspected Runs)

### 5.1 Current Re-Evaluated Failing Trials (n=7)

#### `[single_hop] task-03_t2` (FAIL) - `stopped early` | ReAct: `premature stop`
- **Question:** Which MCP tool deletes multiple dashboards at once in PostHog, and what is the maximum number of dashboards that can be deleted in one request?
- **Trajectory Metrics:** Status=`abstain` | Steps=`2` | Tool Calls=`2` | Duplicate Queries=`0` | Tokens In=`8,796` | Tokens Out=`671` | Latency=`8165.2ms` | Chunks=`7`
- **Citations & Abstain:** Abstain=`True` | Valid=`0` (None) | Hallucinated=`0` (None)
- **Search Queries:** `MCP tool delete multiple dashboards at once bulk delete maximum` -> `PostHog MCP dashboards bulk delete tool`
- **Secondary Signals:** `bad query`
- **Grader Failure Reasons:** `Outcome failed: Agent abstained on an answerable task`; `Outcome failed: No citations provided for answerable task`
- **Transcript Evidence:** Executed 2 parallel search_docs calls in step 1 (total_tool_calls=2 of MAX_STEPS=6, tokens_in=8,796). Call 1 ('MCP tool delete multiple dashboards at once bulk delete maximum') appended 'maximum', which pushed product-analytics/surfaces/mcp.mdx#5 out of top-5 (unlike task-03_t1/task-03_t3 where Call 1 omitted 'maximum' and retrieved mcp.mdx#5). Call 2 ('PostHog MCP dashboards bulk delete tool') also missed mcp.mdx#5. Despite 4 remaining tool-budget steps, the agent abstained immediately at step 2.
- **Root Cause:** Query-term dilution in Call 1 dropped mcp.mdx#5 from top-5, and the agent stopped prematurely at 2 tool calls (4 unused budget steps) with a false abstention instead of reformulating.

#### `[multi_hop] task-10_t1` (FAIL) - `fabricated citation` | ReAct: `context rot`
- **Question:** Before launching a PostHog experiment, how can you test a specific variant without changing code, and what inputs determine how long the experiment must run?
- **Trajectory Metrics:** Status=`completed_with_hallucinated_citations` | Steps=`3` | Tool Calls=`3` | Duplicate Queries=`0` | Tokens In=`17,133` | Tokens Out=`989` | Latency=`10656.8ms` | Chunks=`15`
- **Citations & Abstain:** Abstain=`False` | Valid=`6` (`experiments/no-code-web-experiments.md#7`, `experiments/troubleshooting.mdx#6`, `experiments/sample-size-running-time.mdx#4`, `experiments/sample-size-running-time.mdx#5`, `experiments/sample-size-running-time.mdx#6`, `experiments/best-practices.mdx#4`) | Hallucinated=`1` (`experiments/testing-and-launching.mdx#3`)
- **Search Queries:** `preview experiment variant without code changes` -> `experiment duration calculation minimum detectable effect sample size` -> `test a specific experiment variant before launch feature flag override for myself`
- **Secondary Signals:** `excessive tool calls`
- **Grader Failure Reasons:** `Constraint failed: Hallucinated citations detected: ['experiments/testing-and-launching.mdx#3']`
- **Transcript Evidence:** Executed 3 steps / 3 tool calls accumulating 15 unique chunks and 17,133 input tokens (2.32x task-10_t3's 7,377 tokens). Call 3 retrieved 'experiments/testing-and-launching.mdx#0' at display heading '### [3] Testing and launching an experiment'. In final_output.citations, the model emitted 'experiments/testing-and-launching.mdx#3', conflating rank index [3] with chunk suffix #0.
- **Root Cause:** Under 17,133 input tokens and 15 accumulated chunks across 3 steps, the agent fabricated chunk suffix #3 for retrieved chunk experiments/testing-and-launching.mdx#0.

#### `[unanswerable] task-11_t1` (FAIL) - `over-helpfulness / failed abstain` | ReAct: `failed abstention / over-helpfulness`
- **Question:** Does PostHog provide a native telematics protocol adapter to read raw CAN bus frames directly from automotive ECUs?
- **Trajectory Metrics:** Status=`completed` | Steps=`2` | Tool Calls=`2` | Duplicate Queries=`0` | Tokens In=`7,748` | Tokens Out=`447` | Latency=`7220.0ms` | Chunks=`9`
- **Citations & Abstain:** Abstain=`False` | Valid=`3` (`how-posthog-works/ingestion-pipeline.mdx#0`, `metrics/architecture.mdx#1`, `data-pipelines/sources/npm-registry.md#1`) | Hallucinated=`0` (None)
- **Search Queries:** `CAN bus telematics raw frames ECU ingestion` -> `PostHog data ingestion sources supported protocols`
- **Secondary Signals:** `fabricated citation (unanswerable policy)`
- **Grader Failure Reasons:** `Constraint failed: Fabricated 3 citations on unanswerable task`; `Outcome failed: Expected agent to abstain on unanswerable task, but agent answered`; `Outcome failed: Unanswerable task must not cite any URLs`; `Outcome failed: Unanswerable answer did not express lack of support or documentation`
- **Transcript Evidence:** Unanswerable CAN bus telematics query (steps=2, calls=2, tokens_in=7,748). Retrieved general ingestion pipeline docs (how-posthog-works/ingestion-pipeline.mdx#0, metrics/architecture.mdx#1, npm-registry.md#1). Output began 'No. The documentation I searched contains no mention...' and proposed a custom bridge/OTLP workaround with abstain=false and 3 citations.
- **Root Cause:** Model treated a capability-absence yes/no question as answerable ('No' + custom bridge advice), setting abstain=false with 3 citations and failing unanswerable policy checks (including substring abstain_cues in semantic_correctness).

#### `[unanswerable] task-11_t3` (FAIL) - `over-helpfulness / failed abstain` | ReAct: `failed abstention / over-helpfulness`
- **Question:** Does PostHog provide a native telematics protocol adapter to read raw CAN bus frames directly from automotive ECUs?
- **Trajectory Metrics:** Status=`completed` | Steps=`2` | Tool Calls=`2` | Duplicate Queries=`0` | Tokens In=`6,510` | Tokens Out=`542` | Latency=`7483.5ms` | Chunks=`10`
- **Citations & Abstain:** Abstain=`False` | Valid=`4` (`how-posthog-works/ingestion-pipeline.mdx#0`, `metrics/architecture.mdx#1`, `how-posthog-works/index.mdx#1`, `data-pipelines/sources/mono.md#2`) | Hallucinated=`0` (None)
- **Search Queries:** `CAN bus telematics adapter raw frames ECU ingestion` -> `PostHog data sources import automotive vehicle data`
- **Secondary Signals:** `fabricated citation (unanswerable policy)`
- **Grader Failure Reasons:** `Constraint failed: Fabricated 4 citations on unanswerable task`; `Outcome failed: Expected agent to abstain on unanswerable task, but agent answered`; `Outcome failed: Unanswerable task must not cite any URLs`; `Outcome failed: Unanswerable answer did not express lack of support or documentation`
- **Transcript Evidence:** Unanswerable CAN bus telematics query (steps=2, calls=2, tokens_in=6,510). Retrieved general ingestion architecture and SaaS connector chunks. Answered 'No. The PostHog documentation I searched does not describe any native telematics protocol adapter...' and advised writing custom SDK/OTLP software, setting abstain=false with 4 citations.
- **Root Cause:** Over-helpfulness on negative capability question caused abstain=false and 4 cited general-ingestion chunks (contrast with task-11_t2, which missed ingestion-pipeline.mdx#0 and abstained cleanly).

#### `[unanswerable] task-12_t1` (FAIL) - `over-helpfulness / failed abstain` | ReAct: `failed abstention / over-helpfulness`
- **Question:** Does PostHog offer a native COBOL SDK or z/OS batch library for streaming tracking events from IBM mainframes?
- **Trajectory Metrics:** Status=`completed` | Steps=`3` | Tool Calls=`3` | Duplicate Queries=`0` | Tokens In=`16,005` | Tokens Out=`721` | Latency=`9344.7ms` | Chunks=`15`
- **Citations & Abstain:** Abstain=`False` | Valid=`4` (`health-checks/sdk-health.mdx#4`, `libraries/godot/index.mdx#0`, `web-analytics/sending-http-logs.mdx#1`, `mcp-analytics/installation/other-languages.mdx#0`) | Hallucinated=`0` (None)
- **Search Queries:** `COBOL SDK PostHog mainframe z/OS` -> `PostHog SDKs list supported languages` -> `send events via HTTP capture API without an SDK`
- **Secondary Signals:** `excessive tool calls`, `fabricated citation (unanswerable policy)`
- **Grader Failure Reasons:** `Constraint failed: Fabricated 4 citations on unanswerable task`; `Outcome failed: Expected agent to abstain on unanswerable task, but agent answered`; `Outcome failed: Unanswerable task must not cite any URLs`; `Outcome failed: Unanswerable answer did not express lack of support or documentation`
- **Transcript Evidence:** Unanswerable COBOL/z/OS mainframe query (steps=3, calls=3, tokens_in=16,005, 15 retrieved chunks). After Call 1/2 confirmed COBOL is absent from supported SDKs (health-checks/sdk-health.mdx#4), the agent issued a 3rd search ('send events via HTTP capture API without an SDK') to propose a custom HTTP integration, outputting abstain=false and 4 citations.
- **Root Cause:** Model spent an extra 3rd tool call searching HTTP fallback endpoints to offer a workaround for an unsupported mainframe stack, failing abstain_accuracy and unanswerable citation checks.

#### `[unanswerable] task-12_t2` (FAIL) - `over-helpfulness / failed abstain` | ReAct: `failed abstention / over-helpfulness`
- **Question:** Does PostHog offer a native COBOL SDK or z/OS batch library for streaming tracking events from IBM mainframes?
- **Trajectory Metrics:** Status=`completed` | Steps=`2` | Tool Calls=`2` | Duplicate Queries=`0` | Tokens In=`6,935` | Tokens Out=`506` | Latency=`7071.3ms` | Chunks=`10`
- **Citations & Abstain:** Abstain=`False` | Valid=`3` (`health-checks/sdk-health.mdx#4`, `api/capture.mdx#5`, `web-analytics/sending-http-logs.mdx#1`) | Hallucinated=`0` (None)
- **Search Queries:** `COBOL SDK mainframe z/OS batch event tracking` -> `PostHog SDKs list supported languages`
- **Secondary Signals:** `fabricated citation (unanswerable policy)`
- **Grader Failure Reasons:** `Constraint failed: Fabricated 3 citations on unanswerable task`; `Outcome failed: Expected agent to abstain on unanswerable task, but agent answered`; `Outcome failed: Unanswerable task must not cite any URLs`; `Outcome failed: Unanswerable answer did not express lack of support or documentation`
- **Transcript Evidence:** Unanswerable COBOL/z/OS mainframe query (steps=2, calls=2, tokens_in=6,935). Retrieved capture API (api/capture.mdx#5) and supported SDK list (health-checks/sdk-health.mdx#4). Answered 'No. The documentation does not describe a native COBOL SDK...' and suggested HTTP POSTs to /i/v0/e/, setting abstain=false with 3 citations.
- **Root Cause:** Answered a negative-capability question with 'No' plus a custom capture API workaround and 3 citations instead of setting abstain=true and citations=[].

#### `[unanswerable] task-12_t3` (FAIL) - `over-helpfulness / failed abstain` | ReAct: `failed abstention / over-helpfulness`
- **Question:** Does PostHog offer a native COBOL SDK or z/OS batch library for streaming tracking events from IBM mainframes?
- **Trajectory Metrics:** Status=`completed` | Steps=`3` | Tool Calls=`3` | Duplicate Queries=`0` | Tokens In=`16,427` | Tokens Out=`679` | Latency=`9395.9ms` | Chunks=`15`
- **Citations & Abstain:** Abstain=`False` | Valid=`2` (`health-checks/sdk-health.mdx#4`, `api/index.mdx#0`) | Hallucinated=`0` (None)
- **Search Queries:** `COBOL SDK PostHog mainframe z/OS` -> `PostHog SDKs list supported languages` -> `PostHog capture API send events over HTTP without SDK`
- **Secondary Signals:** `excessive tool calls`, `fabricated citation (unanswerable policy)`
- **Grader Failure Reasons:** `Constraint failed: Fabricated 2 citations on unanswerable task`; `Outcome failed: Expected agent to abstain on unanswerable task, but agent answered`; `Outcome failed: Unanswerable task must not cite any URLs`; `Outcome failed: Unanswerable answer did not express lack of support or documentation`
- **Transcript Evidence:** Unanswerable COBOL/z/OS mainframe query (steps=3, calls=3, tokens_in=16,427, 15 chunks). Issued a 3rd search ('PostHog capture API send events over HTTP without SDK') after confirming COBOL is missing from sdk-health.mdx#4, then answered 'No.' with custom HTTP advice, abstain=false, and 2 citations.
- **Root Cause:** Repeated the same over-helpfulness pattern as task-12_t1/task-12_t2: extra fallback search for /i/v0/e plus abstain=false and 2 citations, leaving unanswerable Pass^3 at 0.0%.

### 5.2 Historical Grader-Bug Failures Resolved by Synonym Expansion (n=3)

#### `[single_hop] task-05_t1` (PASS) - `grader bug` | ReAct: `grader bug`
- **Question:** How can I rotate the feature flags secure API key in PostHog without causing downtime in production?
- **Trajectory Metrics:** Status=`completed` | Steps=`2` | Tool Calls=`2` | Duplicate Queries=`0` | Tokens In=`7,164` | Tokens Out=`899` | Latency=`8717.4ms` | Chunks=`6`
- **Citations & Abstain:** Abstain=`False` | Valid=`3` (`feature-flags/project-wide-settings.mdx#7`, `feature-flags/project-wide-settings.mdx#8`, `feature-flags/project-wide-settings.mdx#9`) | Hallucinated=`0` (None)
- **Search Queries:** `rotate feature flags secure API key without downtime` -> `feature flags secure API key rotate`
- **Secondary Signals:** None
- **Transcript Evidence:** Single-hop key rotation task (steps=2, calls=2, tokens_in=7,164, 3 valid citations, 0 hallucinated). Answer accurately cited feature-flags/project-wide-settings.mdx#7/#8/#9 and stated 'backup key' and 'delete the backup key'. Failed historical reports/agent_eval.md (0/3) only because expected_facts required exact strings ['secondary key', 'delete old key']; passes under current synonym lists.
- **Root Cause:** Historical grader bug: naive exact-substring matching penalized accurate 'backup key' terminology before eval/agent_tasks.json was updated to synonym lists.

#### `[single_hop] task-05_t2` (PASS) - `grader bug` | ReAct: `grader bug`
- **Question:** How can I rotate the feature flags secure API key in PostHog without causing downtime in production?
- **Trajectory Metrics:** Status=`completed` | Steps=`2` | Tool Calls=`2` | Duplicate Queries=`0` | Tokens In=`7,161` | Tokens Out=`841` | Latency=`8157.0ms` | Chunks=`6`
- **Citations & Abstain:** Abstain=`False` | Valid=`4` (`feature-flags/project-wide-settings.mdx#8`, `feature-flags/project-wide-settings.mdx#7`, `feature-flags/project-wide-settings.mdx#9`, `api/project-secret-api-keys/overview.mdx#0`) | Hallucinated=`0` (None)
- **Search Queries:** `rotate feature flags secure API key without downtime` -> `feature flags secure API key rotation`
- **Secondary Signals:** None
- **Transcript Evidence:** Single-hop key rotation task (steps=2, calls=2, tokens_in=7,161, 4 valid citations, 0 hallucinated). Provided both legacy rotation ('backup key', 'delete the backup key') and secret-key migration steps. Falsely failed in historical reports/agent_eval.md due to exact-string expected_facts mismatch; passes now.
- **Root Cause:** Historical grader bug: identical substring mismatch against 'backup key' / 'delete the backup key' caused a false-negative failure in the recorded benchmark report.

#### `[single_hop] task-05_t3` (PASS) - `grader bug` | ReAct: `grader bug`
- **Question:** How can I rotate the feature flags secure API key in PostHog without causing downtime in production?
- **Trajectory Metrics:** Status=`completed` | Steps=`2` | Tool Calls=`2` | Duplicate Queries=`0` | Tokens In=`7,164` | Tokens Out=`642` | Latency=`7481.9ms` | Chunks=`6`
- **Citations & Abstain:** Abstain=`False` | Valid=`4` (`feature-flags/project-wide-settings.mdx#7`, `feature-flags/project-wide-settings.mdx#8`, `feature-flags/project-wide-settings.mdx#9`, `api/index.mdx#12`) | Hallucinated=`0` (None)
- **Search Queries:** `rotate feature flags secure API key without downtime` -> `feature flags secure API key rotate`
- **Secondary Signals:** None
- **Transcript Evidence:** Single-hop key rotation task (steps=2, calls=2, tokens_in=7,164, 4 valid citations, 0 hallucinated). Explained primary/backup key transition and 'delete the backup key'. Falsely failed in historical reports/agent_eval.md under rigid substring matching; passes under current synonym-list grading.
- **Root Cause:** Historical grader bug: completed the 3/3 false-negative failure on task-05 in reports/agent_eval.md despite accurate, fully grounded answers.

### 5.3 Flaky Multi-Step & Standalone Trajectory Runs (n=4)

#### `[single_hop] task-06_t2` (PASS) - `bad query` | ReAct: `excessive tool calls`
- **Question:** Before launching an A/B test in PostHog, what should I verify regarding exposure events, and when does an exposure event fire?
- **Trajectory Metrics:** Status=`completed` | Steps=`3` | Tool Calls=`3` | Duplicate Queries=`0` | Tokens In=`15,185` | Tokens Out=`984` | Latency=`10866.6ms` | Chunks=`12`
- **Citations & Abstain:** Abstain=`False` | Valid=`7` (`experiments/exposures.mdx#16`, `experiments/exposures.mdx#1`, `experiments/exposures.mdx#5`, `experiments/running-experiments-without-feature-flags.mdx#13`, `experiments/running-experiments-without-feature-flags.mdx#3`, `experiments/troubleshooting.mdx#9`, `libraries/godot/index.mdx#20`) | Hallucinated=`0` (None)
- **Search Queries:** `A/B test exposure event when does it fire` -> `experiments pre-launch checklist exposure event verify` -> `exposure event $feature_flag_called fires when feature flag is evaluated experiments`
- **Secondary Signals:** `context growth`
- **Transcript Evidence:** Single-hop exposure event task that passed final grading, but required 3 steps / 3 calls and 15,185 input tokens (2.22x task-06_t1's 6,840 tokens). Call 1 and Call 2 already retrieved experiments/exposures.mdx#10/#14/#16, yet the agent issued a 3rd search for '$feature_flag_called' definition details, duplicating troubleshooting.mdx#10 and running-experiments-without-feature-flags.mdx#3.
- **Root Cause:** Flaky step-count escalation (2 -> 3 calls) on a single-hop task more than doubled input tokens (6,840 -> 15,185) due to broad initial query framing.

#### `[single_hop] task-08_t1` (PASS) - `bad query` | ReAct: `excessive tool calls`
- **Question:** How can I control the percentage of sessions recorded by PostHog Session Replay to manage volume?
- **Trajectory Metrics:** Status=`completed` | Steps=`3` | Tool Calls=`3` | Duplicate Queries=`0` | Tokens In=`20,044` | Tokens Out=`807` | Latency=`9832.3ms` | Chunks=`12`
- **Citations & Abstain:** Abstain=`False` | Valid=`3` (`session-replay/how-to-control-which-sessions-you-record.mdx#3`, `session-replay/how-to-control-which-sessions-you-record.mdx#1`, `session-replay/troubleshooting.mdx#9`) | Hallucinated=`0` (None)
- **Search Queries:** `session replay sampling percentage of sessions recorded` -> `session replay manage volume limit recordings sample rate` -> `legacy controls session recording sample rate posthog-js sessionRecording`
- **Secondary Signals:** `context rot risk`
- **Transcript Evidence:** Highest input-token trajectory in all 36 eval trials (tokens_in=20,044, steps=3, calls=3, 12 unique chunks across 15 retrieved results; 2.30x task-08_t2's 8,725 tokens). Call 2 ('session replay manage volume limit recordings sample rate') missed how-to-control-which-sessions-you-record.mdx completely (pulling replay-vision, dashboards, billing), forcing a 3rd query.
- **Root Cause:** Off-target Call 2 query wasted a retrieval turn and inflated context to 20,044 input tokens across 12 unique chunks (15 retrieved results) before recovering on Call 3.

#### `[multi_hop] task-10_t2` (PASS) - `bad query` | ReAct: `excessive tool calls`
- **Question:** Before launching a PostHog experiment, how can you test a specific variant without changing code, and what inputs determine how long the experiment must run?
- **Trajectory Metrics:** Status=`completed` | Steps=`3` | Tool Calls=`3` | Duplicate Queries=`0` | Tokens In=`15,194` | Tokens Out=`1,285` | Latency=`11806.0ms` | Chunks=`15`
- **Citations & Abstain:** Abstain=`False` | Valid=`7` (`experiments/no-code-web-experiments.md#7`, `experiments/troubleshooting.mdx#6`, `experiments/sample-size-running-time.mdx#5`, `experiments/sample-size-running-time.mdx#4`, `experiments/sample-size-running-time.mdx#3`, `experiments/sample-size-running-time.mdx#6`, `experiments/best-practices.mdx#4`) | Hallucinated=`0` (None)
- **Search Queries:** `preview experiment variant without code changes` -> `experiment duration calculation sample size minimum detectable effect baseline` -> `force a specific feature flag variant for testing experiment`
- **Secondary Signals:** `context growth`
- **Transcript Evidence:** Multi-hop task that passed final grading, but escalated to 3 steps / 3 calls and 15,194 input tokens (2.06x task-10_t3's 7,377 tokens). Call 3 ('force a specific feature flag variant for testing experiment') retrieved 4 off-topic SDK chunks (godot, roblox, vscode-extension, ruby) alongside experiments/troubleshooting.mdx#6.
- **Root Cause:** Under-specified Call 3 query ('force ... variant') pulled 4 irrelevant SDK docs and doubled context size relative to task-10_t3.

#### `[standalone] 2c815ecfbe79` (STANDALONE) - `bad query` | ReAct: `excessive tool calls`
- **Transcript Source:** standalone trial `2c815ecfbe79` (from `runs/transcripts/agent_trials.jsonl`)
- **Question:** how to capture LLM traces and generations in python
- **Trajectory Metrics:** Status=`completed` | Steps=`3` | Tool Calls=`4` | Duplicate Queries=`0` | Tokens In=`18,231` | Tokens Out=`1,643` | Latency=`14518.9ms` | Chunks=`19`
- **Citations & Abstain:** Abstain=`False` | Valid=`7` (`ai-observability/basics.mdx#1`, `ai-observability/generations.mdx#1`, `ai-observability/large-events.mdx#1`, `ai-observability/large-events.mdx#2`, `ai-observability/errors.mdx#5`, `ai-observability/calculating-costs.mdx#7`, `ai-observability/basics.mdx#6`) | Hallucinated=`0` (None)
- **Search Queries:** `how to capture LLM traces and generations in Python` -> `PostHog LLM observability Python SDK generation event $ai_generation` -> `AI Observability installation Python SDK wrapper OpenAI posthog-python setup` -> `manually capture $ai_generation $ai_trace event Python properties $ai_model $ai_input $ai_output_choices`
- **Secondary Signals:** `incomplete retrieval`, `context growth`
- **Transcript Evidence:** Standalone trial 2c815ecfbe79 (from runs/transcripts/agent_trials.jsonl; question: 'how to capture LLM traces and generations in python') ran 3 steps / 4 tool calls, 18,231 input tokens, and 19 retrieved chunks. Call 3 ('AI Observability installation Python SDK wrapper OpenAI posthog-python setup') missed ai-observability/installation.mdx entirely, forcing the final answer to admit it lacked wrapper installation steps.
- **Root Cause:** Keyword-overloaded Call 3 query missed the canonical installation guide even after 4 tool calls and 18,231 input tokens, leaving part of the user's question unfulfilled.

---

## 6. Prioritized Remediation Hypotheses for Agent Release Gates (Top 3)

### Hypothesis 1 (Priority 1 - Highest Release Impact): Negative-Question Abstention Calibration & Deterministic Unanswerable Guard
- **Target Failures:** `task-11_t1`, `task-11_t3`, `task-12_t1`, `task-12_t2`, `task-12_t3` (`5/7` current failures).
- **Root Mechanism:** `src/agent.py` `SYSTEM_PROMPT` tells the agent to abstain when documentation lacks enough information, but provides no negative-capability examples. When users ask whether PostHog natively supports an out-of-domain feature (CAN bus, COBOL), the model answers 'No' with workaround suggestions and valid citations for general APIs (`abstain=false`).
- **Proposed Fix:** (1) Add explicit few-shot negative-capability calibration to `SYSTEM_PROMPT` instructing that when a queried native integration/SDK/adapter does not exist in PostHog docs, the agent MUST set `abstain: true` and `citations: []` without proposing unsupported workarounds. (2) Add a deterministic post-retrieval relevance gate / citation verifier when max cross-encoder score across all tool calls falls below threshold or query entities (e.g., `COBOL`, `CAN bus`) have zero matches in retrieved chunks.
- **Expected Impact:** Lifts unanswerable `Pass^3` from `0.0%` toward `100.0%`, projected to raise overall suite `Pass@1` from `80.6%` (`29/36`) to `94.4%` (`34/36`).

### Hypothesis 2 (Priority 2 - Answerable Reliability & Grounding): Verbatim Chunk-ID Verification & Budget-Aware Reformulation
- **Target Failures:** `task-03_t2` (`stopped early`), `task-10_t1` (`fabricated citation`), plus `4` high-token multi-call runs (`task-06_t2`, `task-08_t1`, `task-10_t2`, standalone trial `2c815ecfbe79` (from `runs/transcripts/agent_trials.jsonl`)).
- **Root Mechanism:** Result headers (`### [N] Title` followed by `- **Chunk ID**: ...`) coupled with elevated multi-step input token load (mean `16,451.0` for 3+ calls vs `7,185.3` for 1-2 calls) induce rank-index conflation (`#3` instead of `#0`), while single-round query misses trigger premature abstention below `MAX_STEPS` (`6`).
- **Proposed Fix:** (1) On `hallucinated_citations` in `src/agent.py`, run a single deterministic repair/retry prompt listing exact `all_retrieved_chunk_ids` (or strip rank numbers `[N]` / format as `<chunk id="...">`). (2) Add a budget-aware reflection guard before permitting `abstain=true` when `total_tool_calls <= 2`, requiring at least one keyword-broadened reformulation before concluding documentation is absent.
- **Expected Impact:** Eliminates answerable citation fabrication and premature false abstention, lifting `single_hop` and `multi_hop` `Pass^3` toward `100.0%`.

### Hypothesis 3 (Priority 3 - Evaluation Hygiene & Gate Synchronization): Semantic Outcome Grading & Automated Report Sync
- **Target Failures:** Historical `task-05` false negatives (`3/10`), brittle `abstain_cues` in `eval/eval_agent.py`, and historical `reports/agent_eval.md` snapshot drift.
- **Root Mechanism:** `grade_semantic_correctness` relies on naive substring matching (`expected_facts` and 9 hardcoded `abstain_cues` missing `'does not mention'` / `'does not describe'` / `'does not list'`), and `reports/agent_eval.md` reflects an earlier task-set snapshot prior to synonym expansion and unanswerable-task updates in `eval/agent_tasks.json`.
- **Proposed Fix:** (1) Expand `abstain_cues` or route borderline semantic checks through a calibrated LLM judge while preserving deterministic trajectory constraint graders. (2) Enforce a CI/regression check verifying `reports/agent_eval.md` matches deterministic re-evaluation of saved transcripts whenever `eval/agent_tasks.json` or `eval/eval_agent.py` changes.
- **Expected Impact:** Prevents false-negative grader drift and ensures automated CI/eval summaries stay synchronized with the committed task suite.

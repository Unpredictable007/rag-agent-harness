# Agent Evaluation Benchmark Report (Milestone M10)

## Executive Summary

- **Model**: `claude-haiku-5-5`
- **Total Tasks**: 12 (8 single-hop, 2 multi-hop, 2 unanswerable)
- **Trials per Task**: 3 (Total Trials: 36)
- **Overall Pass@1**: 72.2%
- **Overall Pass@3 ($pass@3$)**: 83.3%
- **Overall Pass^3 ($pass^3$)**: 58.3%
- **Total Benchmark Duration**: 445.2s

### Key High-Level Metrics

| Metric | Definition | Value |
| :--- | :--- | :--- |
| **Pass@1** | Mean trial success rate across all 36 trials | 72.2% |
| **Pass@3 ($pass@3$)** | Task success rate where at least 1 of 3 trials succeeds (capability/coverage) | 83.3% |
| **Pass^3 ($pass^3$)** | Task success rate where all 3 of 3 trials succeed (consistency/reliability) | 58.3% |

---

## Task Type Breakdown

| Task Type | Task Count | Pass@1 | Pass@3 ($pass@3$) | Pass^3 ($pass^3$) |
| :--- | :---: | :---: | :---: | :---: |
| `single_hop` | 8 | 83.3% | 87.5% | 75.0% |
| `multi_hop` | 2 | 83.3% | 100.0% | 50.0% |
| `unanswerable` | 2 | 16.7% | 50.0% | 0.0% |

---

## Grader Performance Breakdown

### Trajectory Constraint Graders (Deterministic, $0 Cost)

Trajectory constraints verify the agent's behavior and protocol discipline without imposing brittle tool sequence requirements:

| Constraint | Description | Pass Rate |
| :--- | :--- | :---: |
| `constraint_search_called` | >= 1 search call on answerable queries | 100.0% |
| `constraint_multihop_depth` | >= 2 distinct searches on multi-hop tasks | 100.0% |
| `constraint_step_budget` | Trajectory finished within MAX_STEPS (6) | 100.0% |
| `constraint_no_duplicate_queries` | Loop guard prevented redundant queries | 100.0% |
| `constraint_no_fabricated_citations` | Zero hallucinated or fabricated citations | 83.3% |

### Outcome Graders

Outcome graders verify that the final answer is truthful, well-grounded, and correctly abstains when information is absent:

| Grader | Description | Pass Rate |
| :--- | :--- | :---: |
| `abstain_accuracy` | Abstains if and only if question is unanswerable | 83.3% |
| `citation_accuracy` | Cited passages resolve to verified gold URLs | 83.3% |
| `semantic_correctness` | Answer accurately reflects key documented facts | 75.0% |

---

## Detailed Per-Task Trial Results

| ID | Type | Answerable | Successes | Pass@3 | Pass^3 | Failures / Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `task-01` | `single_hop` | Yes | 3/3 | PASS | PASS | All trials passed cleanly |
| `task-02` | `single_hop` | Yes | 3/3 | PASS | PASS | All trials passed cleanly |
| `task-03` | `single_hop` | Yes | 2/3 | PASS | FAIL | Outcome failed: Agent abstained on an answerable task; Outcome failed: No citations provided for answerable task |
| `task-04` | `single_hop` | Yes | 3/3 | PASS | PASS | All trials passed cleanly |
| `task-05` | `single_hop` | Yes | 0/3 | FAIL | FAIL | Outcome failed: Matched 0/2 expected facts (needed 1). Missing: ['secondary key', 'delete old key'] |
| `task-06` | `single_hop` | Yes | 3/3 | PASS | PASS | All trials passed cleanly |
| `task-07` | `single_hop` | Yes | 3/3 | PASS | PASS | All trials passed cleanly |
| `task-08` | `single_hop` | Yes | 3/3 | PASS | PASS | All trials passed cleanly |
| `task-09` | `multi_hop` | Yes | 3/3 | PASS | PASS | All trials passed cleanly |
| `task-10` | `multi_hop` | Yes | 2/3 | PASS | FAIL | Constraint failed: Hallucinated citations detected: ['experiments/testing-and-launching.mdx#3'] |
| `task-11` | `unanswerable` | No | 1/3 | PASS | FAIL | Constraint failed: Fabricated 3 citations on unanswerable task; Outcome failed: Expected agent to abstain on unanswerable task, but agent answered |
| `task-12` | `unanswerable` | No | 0/3 | FAIL | FAIL | Constraint failed: Fabricated 4 citations on unanswerable task; Outcome failed: Expected agent to abstain on unanswerable task, but agent answered |

---

## Analysis: Why Pass@3 and Pass^3 Differ

A critical finding from Anthropic's 'Demystifying evals' framework is the fundamental operational distinction between $pass@k$ and $pass^k$:

1. **Capability vs Reliability Frontier**:
   - **$pass@3$ (Capability/Coverage)** measures whether the agent is capable of finding the right tool calls and answers given multiple independent attempts. A high $pass@3$ indicates strong reasoning capacity and broad coverage.
   - **$pass^3$ (Reliability/Consistency)** measures whether the agent produces correct, grounded results 100% of the time without failing any trajectory constraints or hallucinating citations.

2. **Stochastic Trajectory Branching**:
   - Even at low sampling temperature, LLM tool selection exhibits non-deterministic branching. On complex multi-hop tasks, trial 1 might query an optimal keyword on step 1, whereas trial 2 might issue a slightly broader query requiring follow-up retrieval.
   - When multi-hop depth requires combining two distinct documents, occasional single-search shortcuts cause variance between individual trials.

3. **Poka-Yoke Guards Preserve Safety**:
   - The duplicate query loop guard successfully prevents infinite loops when queries repeat, allowing the agent to recover or conclude safely.
   - Step budget clamping prevents context overflow, converting runaway searches into safe fallback conclusions.

## Remediation Analysis: Addressing 0% Pass^3 on Unanswerable Tasks

The benchmark revealed that unanswerable tasks represent the steepest reliability bottleneck (`Pass^3 = 0.0%`, `Pass@1 = 16.7%`). Both failure modes stemmed from model over-helpfulness: on `task-11` (IBM z/OS mainframe) and `task-12` (quantum telemetry SLA), Claude Haiku attempted to synthesize an answer from parametric memory or tangential search results rather than abstaining, leading to hallucinated citations and failed abstain checks.

In production agent harnesses, four targeted remediations resolve this failure mode:

1. **Negative Calibration & Few-Shot Abstention in System Prompt**:
   - Instruction: Current models suffer from an alignment bias toward helpfulness. Incorporating explicit few-shot examples showing how to abstain when retrieved documentation lacks direct relevance signals to the model that abstaining is a high-reward behavior.
   - Poka-yoke rule: *"If retrieved passages do not explicitly discuss the queried system or feature, output abstain=true with an empty citations list. Do NOT attempt general guidance."*

2. **Cross-Encoder Score Relevance Gating (Retrieval Confidence Threshold)**:
   - Automated cutoff: If the maximum reranker score among all retrieved passages falls below a calibrated threshold (e.g. score <= 0.40), the harness can automatically prepend a low-confidence warning or trigger a deterministic fallback, preventing the model from over-interpreting low-similarity chunks.

3. **Post-Generation Hallucination / Citation Guard (Deterministic)**:
   - Verifier: An outcome filter that verifies whether cited chunk IDs actually contain semantic tokens matching the question keywords. If an agent attempts to cite irrelevant documents while generating an answer for an unanswerable topic, the harness overrides the output to abstain.

4. **Dedicated Abstention Classifier (Two-Stage Architecture)**:
   - Classifier: For critical enterprise domains, a lightweight binary classifier (or calibrated small LLM) determines answerability based on query and retrieved context before generating a user-facing response.

# CLAUDE.md — Build Spec & Project Context

This file is project context for Claude Code. It describes **what to build, in what order, and the constraints that are easy to get wrong.** Read it before writing code in this repo.

---

## What this project is (and is not)

**Is:** RL post-training of a RAG generator to be faithful to retrieved evidence, in the **RLAIF** family. A single groundedness **verifier** drives two training paths — **DPO** (fast baseline) and **GRPO** (reward-based, where reward hacking is studied). Domain: factual QA over mental-health clinical reference text. Retrieval stack: **Cohere Embed + Rerank**.

**Is NOT:**
- Not a chatbot, not therapy, not companionship dialogue. No multi-turn emotional support. The task is single-shot factual QA grounded in sources.
- Not a clinical tool. No diagnosis, no individualized advice. Filter out "what should I do about my symptoms" questions and **all self-harm / suicide content** at the data stage.
- Not strict RLVR. The groundedness signal is a learned NLI model. Only `copy_penalty` is programmatically verifiable. Do not label this RLVR anywhere in code, comments, or docs.

---

## Architecture: one verifier, two paths

The verifier is the foundation and is **built and calibrated before any training**. It is reused, not rebuilt, for each path:
- DPO uses it to auto-label answers into `chosen` / `rejected` pairs.
- GRPO uses its score as a scalar reward.

Do not implement two separate "graders." There is **one** `verifier/` module; both training paths import it.

---

## Build order (do not reorder)

### Foundation
1. **`eval/` first.** Define metrics before building anything to optimize:
   - `groundedness_rate` — RAGAS-style: (# supported claims) / (# total claims).
   - `citation_precision`, `citation_recall`.
   - `copy_rate` — n-gram overlap between answer and its cited passages (hacking probe).
   - `abstention_probe` — detect "I don't know"/evasive answers (collapse probe; cf. MTRAG IDK judge).
   - `helpfulness` — LLM-as-judge (used later, in GRPO act).
2. **Data filtering (`data/`).** Keep only factual items: diagnostic criteria, therapy-technique definitions, symptom descriptions. **Drop:** personal-advice questions, anything individualized, anything self-harm-related. Use openly-licensed sources (ICD-11, open textbooks, CC-BY summaries). **Do not commit DSM-5 or other copyrighted full text.**
3. **`retrieval/` baseline RAG.** Cohere Embed → Cohere Rerank → small generator answering from top-k passages. Measure and record baseline `groundedness_rate` — this is the number both paths must beat.
4. **`verifier/`.** Input `(retrieved_passages, answer)` → groundedness scalar. MVP = whole-answer granularity: passages as premise, answer as hypothesis, NLI model (DeBERTa-NLI or Vectara HHEM) → `score = P_entail − P_contradiction`. This is **entailment, not cosine similarity** — do not use embedding similarity as the groundedness signal.
5. **`verifier/` calibration (gate).** Hand-label 30–50 `(passage, answer, grounded?)` examples (binary for MVP). Compute agreement between verifier and human labels. **If agreement is low, do not proceed to training** — either swap to a small-LLM judge or narrow the corpus to harder-fact items, then re-test. This gate protects both paths; DPO is *more* sensitive to verifier noise than GRPO, so do not skip it.

### Act 1 — DPO (`training/dpo/`)
6. **Preference-pair construction.** Sample multiple answers per question from the baseline; score with the verifier; high → `chosen`, low → `rejected`. **Hard negatives matter:** `rejected` should be *plausible-but-unsupported* (e.g. grounded answer with one injected claim absent from the source), not random bad text. Otherwise DPO won't learn the faithful-vs-plausible boundary.
7. **Train DPO** via TRL. Report `groundedness_rate` vs baseline. This is the shippable safety-net version.

### Act 2 — GRPO (`training/grpo/`)
8. **Reward function.** Start with `R = groundedness − λ·copy_penalty`. `groundedness` from verifier (learned); `copy_penalty` = n-gram overlap (programmatic). **Intentionally omit helpfulness at first** — the goal is to observe collapse, not prevent it.
9. **GRPO loop** via TRL. Train, watch groundedness rise while answers degrade (verbatim copying or terse/evasive output).
10. **Diagnose + fix.** Identify the collapse/hacking mode, add `+ μ·helpfulness` (LLM-judge), retrain. Keep **before vs after** artifacts — this is the project's most valuable result.

### Wrap-up
11. **`eval/` comparison.** baseline vs DPO vs GRPO on all metrics. Save failure-case examples and the verifier calibration agreement number.

---

## Key constraints & gotchas

- **Use TRL for both DPO and GRPO.** Do not hand-roll RL. Effort goes into reward design, data, and eval — not the optimizer.
- **GRPO reward stability:** `P_entail − P_contradiction` fed directly to GRPO can be noisy (NLI is jumpy on long premises). If reward is unstable, clip or normalize. GRPO's group baseline already helps; this is *why* GRPO over PPO here.
- **Sparse reward:** whole-answer granularity gives one score per answer — fine for MVP, but for *diagnosing which sentence is unfaithful* you'll need sentence-level claim decomposition. Treat sentence-level as a stretch upgrade.
- **Generator size:** keep to 1–3B so DPO and GRPO fit on limited GPU. LoRA/PEFT for training.
- **Never** put the verifier's similarity (cosine) where its entailment judgment belongs. Similarity is for retrieval; entailment is for faithfulness.
- **Package install:** `pip install --break-system-packages` in this environment.

---

## Definition of done (per stage)

- Foundation done = baseline groundedness recorded + verifier calibration agreement ≥ ~85% (or corpus/judge adjusted until it is).
- Act 1 done = DPO version beats baseline on groundedness; preference-pair pipeline reproducible.
- Act 2 done = at least one reward-hacking/collapse mode found, documented, and repaired with before/after numbers.

When unsure whether something belongs in scope, check it against "Is NOT" at the top. If it edges toward dialogue, advice, diagnosis, or self-harm content — it's out.
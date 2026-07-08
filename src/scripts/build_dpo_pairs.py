"""Build DPO preference pairs from the baseline generator + calibrated judge.

CLAUDE.md step 6. For each question:
  1. Retrieve correct passages.
  2. Sample K candidate answers on the correct passages (temperature sampling).
  3. Also sample H "hard negative" candidates using passages retrieved for a
     *different* question — plausible-but-unsupported, exactly the failure
     mode DPO must learn to reject. Passage-swap is the same trick the
     calibration pipeline used for hard negatives.
  4. Score every candidate with the LLM-judge verifier on the correct passages.
  5. Emit (prompt, chosen, rejected) triples where the score gap is large
     enough that the pair is unambiguous.

The judge is expensive-ish (Cohere API), so an audit trail of every scored
candidate is written to `scored_candidates.jsonl` — re-runs can skip
already-scored (question_id, candidate_idx) tuples if you extend this later.

Usage:
    python -m src.scripts.build_dpo_pairs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml
from dotenv import load_dotenv

from src.grounded_rag.generation.generator import GenerationConfig, HFGenerator
from src.grounded_rag.generation.prompt import Passage, build_messages
from src.grounded_rag.retrieval import Retriever, load_config as load_retrieval_config
from src.grounded_rag.verifier import LLMJudgeConfig, LLMJudgeVerifier


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_questions(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _qid(question: str) -> str:
    return "q:" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]


def _to_passages(retrieved) -> List[Passage]:
    return [Passage(chunk_id=r.chunk_id, title=r.title, text=r.text) for r in retrieved]


def _render_prompt(tokenizer, question: str, passages: Sequence[Passage]) -> str:
    """Render the RAG prompt the way training will see it — chat template
    applied, generation-prompt appended, so DPOTrainer's tokenizer sees the
    same string as inference. Do NOT reimplement here; reuse build_messages."""
    messages = build_messages(question, passages)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _sample_k(generator: HFGenerator, question: str, passages: Sequence[Passage],
              k: int, temperature: float, top_p: float) -> List[str]:
    """K stochastic samples from the generator. We override sampling knobs
    on the cfg for the duration of this batch, then restore."""
    original = replace(generator.cfg)
    generator.cfg.do_sample = True
    generator.cfg.temperature = temperature
    generator.cfg.top_p = top_p
    try:
        # Vary the RNG per-sample so we actually get different completions.
        import torch
        outs = []
        for i in range(k):
            torch.manual_seed(original.seed + i + 1)
            outs.append(generator.generate(question, passages))
        return outs
    finally:
        generator.cfg = original


def _build_verifier(vcfg: dict) -> LLMJudgeVerifier:
    return LLMJudgeVerifier(LLMJudgeConfig(
        model_name=vcfg["judge_model_name"],
        temperature=vcfg["judge_temperature"],
        api_key_env=vcfg["judge_api_key_env"],
    ))


def _pair_from_candidates(cands: List[dict], chosen_min: float,
                          rejected_max: float, min_gap: float) -> Optional[dict]:
    """Pick the highest-score candidate as chosen and lowest as rejected,
    subject to the thresholds. Returns None if no clean pair exists —
    ambiguous pairs are worse than no pairs for DPO."""
    if len(cands) < 2:
        return None
    ranked = sorted(cands, key=lambda c: c["score"], reverse=True)
    chosen, rejected = ranked[0], ranked[-1]
    if chosen["score"] < chosen_min:
        return None
    if rejected["score"] > rejected_max:
        return None
    if chosen["score"] - rejected["score"] < min_gap:
        return None
    return {
        "prompt": chosen["prompt"],
        "chosen": chosen["answer"],
        "rejected": rejected["answer"],
        "chosen_score": chosen["score"],
        "rejected_score": rejected["score"],
        "chosen_source": chosen["source"],
        "rejected_source": rejected["source"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-config", default="configs/dpo.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--generation-config", default="configs/generation.yaml")
    ap.add_argument("--verifier-config", default="configs/verifier.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of questions for a smoke test")
    args = ap.parse_args()

    load_dotenv()

    dcfg = _load_yaml(args.dpo_config)
    retr_cfg = load_retrieval_config(args.retrieval_config)
    gen_raw = _load_yaml(args.generation_config)
    vcfg = _load_yaml(args.verifier_config)

    questions = _load_questions(dcfg["questions_file"])
    if args.limit:
        questions = questions[: args.limit]
    if len(questions) < 2:
        raise SystemExit("Need at least 2 questions for passage-swap hard negatives.")

    retriever = Retriever(retr_cfg)
    generator = HFGenerator(GenerationConfig(**gen_raw))
    generator._load()  # need tokenizer for prompt rendering below
    tokenizer = generator._tokenizer
    judge = _build_verifier(vcfg)

    K = dcfg["samples_per_question"]
    H = dcfg["hard_negatives_per_question"]
    temp = dcfg["sample_temperature"]
    top_p = dcfg["sample_top_p"]

    # Retrieve once per question — the API cost we want to amortize.
    print(f"Retrieving passages for {len(questions)} questions...")
    per_q_passages: Dict[str, List[Passage]] = {}
    per_q_text: Dict[str, str] = {}
    for row in questions:
        q = row["question"]
        qid = _qid(q)
        per_q_passages[qid] = _to_passages(retriever.retrieve(q))
        per_q_text[qid] = q

    qids = list(per_q_passages.keys())
    rng = random.Random(20260707)

    scored_path = Path(dcfg["scored_out"])
    pairs_path = Path(dcfg["pairs_out"])
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored_fh = open(scored_path, "w", encoding="utf-8")
    pairs_fh = open(pairs_path, "w", encoding="utf-8")
    n_pairs = 0

    for qid in qids:
        q = per_q_text[qid]
        passages = per_q_passages[qid]
        prompt = _render_prompt(tokenizer, q, passages)

        # 1. Sample K on correct passages.
        clean_answers = _sample_k(generator, q, passages, K, temp, top_p)

        # 2. Sample H hard negatives on swapped passages (from a different qid).
        hard_answers: List[str] = []
        donor_pool = [x for x in qids if x != qid]
        for _ in range(H):
            donor = rng.choice(donor_pool)
            swapped_passages = per_q_passages[donor]
            hard_answers.extend(
                _sample_k(generator, q, swapped_passages, 1, temp, top_p)
            )

        # 3. Score all on the *correct* passages — the judge must catch that
        #    a swapped-passage answer isn't supported by the true evidence.
        candidates: List[dict] = []
        for i, a in enumerate(clean_answers):
            s = judge.score([p.text for p in passages], a)
            rec = {
                "qid": qid, "question": q, "answer": a, "score": s,
                "source": "clean", "cand_idx": i, "prompt": prompt,
            }
            candidates.append(rec)
            scored_fh.write(json.dumps({k: v for k, v in rec.items() if k != "prompt"}) + "\n")
        for i, a in enumerate(hard_answers):
            s = judge.score([p.text for p in passages], a)
            rec = {
                "qid": qid, "question": q, "answer": a, "score": s,
                "source": "swapped", "cand_idx": K + i, "prompt": prompt,
            }
            candidates.append(rec)
            scored_fh.write(json.dumps({k: v for k, v in rec.items() if k != "prompt"}) + "\n")
        scored_fh.flush()

        # 4. Form a pair if one candidate is clearly better than another.
        pair = _pair_from_candidates(
            candidates,
            chosen_min=dcfg["chosen_min_score"],
            rejected_max=dcfg["rejected_max_score"],
            min_gap=dcfg["min_pair_gap"],
        )
        if pair is None:
            print(f"  {qid}: no clean pair (scores: "
                  f"{sorted(c['score'] for c in candidates)})")
            continue
        pair["qid"] = qid
        pairs_fh.write(json.dumps(pair) + "\n")
        pairs_fh.flush()
        n_pairs += 1
        print(f"  {qid}: chosen={pair['chosen_score']:.2f} "
              f"({pair['chosen_source']}) / "
              f"rejected={pair['rejected_score']:.2f} ({pair['rejected_source']})")

    scored_fh.close()
    pairs_fh.close()
    print(f"\nWrote {n_pairs} pairs → {pairs_path}")
    print(f"Audit trail: {scored_path}")
    if n_pairs < 10:
        print("\nWARN: fewer than 10 pairs — DPO won't have much to learn from. "
              "Consider lowering min_pair_gap, adding more questions, or raising "
              "sample_temperature.")


if __name__ == "__main__":
    main()

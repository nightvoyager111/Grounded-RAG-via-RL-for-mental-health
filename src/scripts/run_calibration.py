"""Score labeled calibration examples with the NLI verifier and report
agreement — the CLAUDE.md step-5 gate.

Reads data/calibration/labeled.jsonl (labels: 0 or 1). Writes
data/calibration/report.json and prints a summary. Gate: agreement ≥ 0.85.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.grounded_rag.verifier import (
    HHEMConfig,
    HHEMVerifier,
    LLMJudgeConfig,
    LLMJudgeVerifier,
    NLIConfig,
    NLIVerifier,
    calibration_report,
    read_labeled,
    score_records,
    write_records,
)


def _build_verifier(raw: dict):
    backend = raw.get("backend", "nli").lower()
    if backend == "llm_judge":
        return LLMJudgeVerifier(LLMJudgeConfig(
            model_name=raw["judge_model_name"],
            temperature=raw["judge_temperature"],
            api_key_env=raw["judge_api_key_env"],
        ))
    if backend == "hhem":
        return HHEMVerifier(HHEMConfig(
            model_name=raw["hhem_model_name"],
            revision=raw.get("hhem_revision"),
            device=raw["device"],
            passage_join=raw["passage_join"],
            aggregate=raw.get("aggregate", "max"),
        ))
    if backend == "nli":
        return NLIVerifier(NLIConfig(
            model_name=raw["model_name"],
            device=raw["device"],
            dtype=raw["dtype"],
            max_length=raw["max_length"],
            passage_join=raw["passage_join"],
            aggregate=raw.get("aggregate", "max"),
        ))
    raise ValueError(f"unknown backend: {backend!r}")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier-config", default="configs/verifier.yaml")
    ap.add_argument("--labeled", default=None,
                    help="override path to labeled JSONL")
    ap.add_argument("--report", default=None,
                    help="override path to report JSON")
    args = ap.parse_args()

    with open(args.verifier_config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    labeled_path = args.labeled or raw["calibration_labeled"]
    report_path = args.report or raw["calibration_report"]

    verifier = _build_verifier(raw)

    records = read_labeled(labeled_path)
    if not records:
        raise SystemExit(f"No records in {labeled_path}")
    labeled = [r for r in records if r.label in (0, 1)]
    if len(labeled) < len(records):
        skipped = len(records) - len(labeled)
        print(f"WARN: skipping {skipped} unlabeled records")
    if not labeled:
        raise SystemExit("No labeled records to score. Set the label field on each row.")

    scored = score_records(labeled, verifier)
    report = calibration_report(scored, default_threshold=raw["threshold"])

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # Also persist scored records so you can inspect per-example scores.
    scored_path = str(Path(report_path).with_name("scored.jsonl"))
    write_records(scored_path, scored)

    print(json.dumps({
        "n_examples": report["n_examples"],
        "positive_rate": report["positive_rate"],
        "at_default_threshold": report["at_default_threshold"],
        "best_threshold": report["best_threshold"],
        "gate_pass_at_default": report["gate_pass_at_default"],
        "gate_pass_at_best": report["gate_pass_at_best"],
    }, indent=2))
    print(f"\nReport: {report_path}\nScored: {scored_path}")

    if not report["gate_pass_at_best"]:
        print("\nGate NOT passed (< 0.85). CLAUDE.md step 5: do not proceed "
              "to training. Try a stronger NLI model, narrow the corpus to "
              "harder-fact items, or switch to a small-LLM judge.")


if __name__ == "__main__":
    main()

"""Interactive terminal labeler for calibration examples.

Reads data/calibration/pending.jsonl, prompts for a binary grounded/not label
one example at a time, and appends each decision to data/calibration/labeled.jsonl
immediately (so Ctrl-C never loses work). Resumable: rows whose id is already
in labeled.jsonl are skipped.

Usage:
    python -m src.scripts.label_calibration
    python -m src.scripts.label_calibration --pending path.jsonl --labeled out.jsonl

Keys:
    y  grounded (label=1)
    n  not grounded (label=0)
    s  skip this example (don't write anything, come back later)
    u  unsure — writes label=null so you can revisit
    b  back — undo the last written label (labeled or unsure)
    q  quit
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rewrite_jsonl(path: Path, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _term_width(default: int = 100) -> int:
    try:
        return max(60, shutil.get_terminal_size().columns)
    except Exception:
        return default


def _hr(char: str = "─") -> str:
    return char * _term_width()


def _wrap(text: str, indent: str = "  ") -> str:
    w = _term_width() - len(indent)
    return textwrap.fill(text, width=w, initial_indent=indent,
                         subsequent_indent=indent, replace_whitespace=False,
                         drop_whitespace=False)


def _render(rec: dict, idx: int, total: int, done: int) -> None:
    print("\n" + _hr("═"))
    print(f"  [{idx + 1}/{total}]   labeled so far: {done}   id: {rec.get('id', '?')}")
    print(_hr("═"))
    print("\n\033[1mQuestion:\033[0m")
    print(_wrap(rec.get("question", "").strip()))
    print("\n\033[1mAnswer:\033[0m")
    print(_wrap(rec.get("answer", "").strip()))
    passages = rec.get("passages", []) or []
    print(f"\n\033[1mPassages ({len(passages)}):\033[0m")
    for i, p in enumerate(passages, 1):
        p = p.strip().replace("\n", " ")
        # Cap each passage preview so long ones don't drown the terminal.
        if len(p) > 700:
            p = p[:700] + " …"
        print(f"\n  \033[2m[{i}]\033[0m")
        print(_wrap(p, indent="      "))
    print("\n" + _hr())


def _prompt() -> str:
    while True:
        try:
            ch = input("  [y] grounded  [n] not  [u] unsure  [s] skip  [b] back  [q] quit  > ").strip().lower()
        except EOFError:
            return "q"
        if ch in {"y", "n", "u", "s", "b", "q"}:
            return ch
        print("  ? use one of: y n u s b q")


def _load_paths(args) -> tuple[Path, Path]:
    if args.pending and args.labeled:
        return Path(args.pending), Path(args.labeled)
    with open(args.verifier_config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return (
        Path(args.pending or raw["calibration_pending"]),
        Path(args.labeled or raw["calibration_labeled"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier-config", default="configs/verifier.yaml")
    ap.add_argument("--pending", default=None)
    ap.add_argument("--labeled", default=None)
    ap.add_argument("--relabel", action="store_true",
                    help="revisit ids already in labeled.jsonl (overwrites them)")
    args = ap.parse_args()

    pending_path, labeled_path = _load_paths(args)
    pending = _read_jsonl(pending_path)
    if not pending:
        raise SystemExit(f"No records in {pending_path}")

    labeled = _read_jsonl(labeled_path)
    by_id: Dict[str, dict] = {r["id"]: r for r in labeled if "id" in r}

    if args.relabel:
        queue = pending
    else:
        queue = [r for r in pending if r["id"] not in by_id or by_id[r["id"]].get("label") is None]

    if not queue:
        print(f"All {len(pending)} examples already labeled in {labeled_path}. "
              f"Re-run with --relabel to revise.")
        return

    print(f"Pending: {pending_path}  ({len(pending)} total)")
    print(f"Output:  {labeled_path}  ({sum(1 for r in labeled if r.get('label') in (0, 1))} labeled)")
    print(f"Queue:   {len(queue)} to review")

    write_history: List[str] = []  # ids written this session, for 'back'
    i = 0
    while i < len(queue):
        rec = queue[i]
        done = sum(1 for r in by_id.values() if r.get("label") in (0, 1))
        _render(rec, i, len(queue), done)
        ch = _prompt()

        if ch == "q":
            print(f"\nStopped at {i}/{len(queue)}. Progress saved to {labeled_path}.")
            return
        if ch == "s":
            i += 1
            continue
        if ch == "b":
            if not write_history:
                print("  (nothing to undo this session)")
                continue
            last_id = write_history.pop()
            by_id.pop(last_id, None)
            _rewrite_jsonl(labeled_path, list(by_id.values()))
            # rewind queue to that record
            for j, r in enumerate(queue):
                if r["id"] == last_id:
                    i = j
                    break
            print(f"  undone: {last_id}")
            continue

        label = 1 if ch == "y" else 0 if ch == "n" else None
        out = dict(rec)
        out["label"] = label
        out["score"] = None

        by_id[out["id"]] = out
        _rewrite_jsonl(labeled_path, list(by_id.values()))
        write_history.append(out["id"])
        i += 1

    done = sum(1 for r in by_id.values() if r.get("label") in (0, 1))
    print(f"\nDone. {done} labeled → {labeled_path}")
    print("Next: python -m src.scripts.run_calibration")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Progress already saved.", file=sys.stderr)
        sys.exit(130)

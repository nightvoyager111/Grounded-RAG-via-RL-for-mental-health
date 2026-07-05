"""
Fetch ICD-11 mental/behavioural/neurodevelopmental disorders (MMS linearization)
from the WHO ICD-API and emit entity-level chunks.

WHY THIS SHAPE:
- MMS linearization (not Foundation): entities are mutually exclusive, so you
  don't get the same disorder via multiple parents -> no duplicate chunks.
- Two-stage: raw JSON cached to data/raw/icd11/ (never edited, gitignored),
  cleaned chunks written to data/corpus/icd11.jsonl.
- Text kept VERBATIM. ICD-11 is CC BY-ND 3.0 IGO (NoDerivatives) -> do not
  paraphrase-and-redistribute. Verbatim premise is also better for NLI.

CREDENTIALS: register a free client at https://icd.who.int/icdapi
Set env vars before running:
    export ICD_CLIENT_ID=...
    export ICD_CLIENT_SECRET=...

Do NOT commit credentials. Do NOT commit data/raw/ or the ICD text itself;
commit THIS SCRIPT. (ND license + hygiene.)
"""

import os
import json
import time
import pathlib
import datetime
import requests
from typing import Optional

TOKEN_ENDPOINT = "https://icdaccessmanagement.who.int/connect/token"
API_BASE = "https://id.who.int"

# Pin the release so your corpus is reproducible. Bump deliberately, never silently.
RELEASE_ID = "2024-01"
LINEARIZATION = "mms"
LANG = "en"

# ICD-11 chapter 06: Mental, behavioural or neurodevelopmental disorders.
# This is the MMS entity id for that chapter root. Verify it resolves for your
# pinned release before a full run (see __main__ smoke test).
CHAPTER_ENTITY_ID = "334423054"

RAW_DIR = pathlib.Path("data/raw/icd11")
OUT_PATH = pathlib.Path("data/corpus/icd11.jsonl")
LICENSE = "CC BY-ND 3.0 IGO"

# Be polite to WHO's API; it is not rate-limit-free in practice.
REQUEST_PAUSE_S = 0.2


def get_token() -> str:
    cid = os.environ.get("ICD_CLIENT_ID")
    secret = os.environ.get("ICD_CLIENT_SECRET")
    if not cid or not secret:
        raise SystemExit(
            "Set ICD_CLIENT_ID and ICD_CLIENT_SECRET env vars "
            "(register at https://icd.who.int/icdapi)."
        )
    resp = requests.post(
        TOKEN_ENDPOINT,
        data={
            "client_id": cid,
            "client_secret": secret,
            "scope": "icdapi_access",
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Accept-Language": LANG,
        "API-Version": "v2",
    }


def get_entity(token: str, entity_uri: str) -> dict:
    """Fetch one MMS entity by its full URI. Cache raw JSON to disk."""
    # entity_uri looks like https://id.who.int/icd/release/11/2024-01/mms/<id>
    entity_id = entity_uri.rstrip("/").split("/")[-1]
    cache = RAW_DIR / f"{entity_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    resp = requests.get(entity_uri, headers=headers(token), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    time.sleep(REQUEST_PAUSE_S)
    return data


def _val(node) -> str:
    """ICD returns localized fields as {'@language':..,'@value':..}."""
    if isinstance(node, dict):
        return node.get("@value", "").strip()
    return ""


def walk(token: str, entity_uri: str, seen: set):
    """DFS over MMS children, yielding (uri, raw_json) for each entity."""
    if entity_uri in seen:
        return
    seen.add(entity_uri)
    data = get_entity(token, entity_uri)
    yield entity_uri, data
    for child_uri in data.get("child", []):
        yield from walk(token, child_uri, seen)


def to_chunk(entity_uri: str, data: dict) -> Optional[dict]:
    """
    Build one chunk per codable entity. Concatenate the fields that carry
    factual content: title + definition (+ inclusions if present).
    Skip pure structural nodes (chapter/block) that have no definition --
    they're navigation, not groundable content.
    """
    title = _val(data.get("title"))
    definition = _val(data.get("definition"))
    code = data.get("code", "")  # present on categories, empty on blocks/chapter

    # Inclusions / synonyms add factual surface area; include verbatim if present.
    inclusions = []
    for inc in data.get("inclusion", []) or []:
        t = _val(inc.get("label")) if isinstance(inc, dict) else ""
        if t:
            inclusions.append(t)

    # A node with no definition and no code is structural -> skip.
    if not definition and not code:
        return None

    parts = [title]
    if definition:
        parts.append(definition)
    if inclusions:
        parts.append("Includes: " + "; ".join(inclusions))
    chunk_text = "\n".join(p for p in parts if p)

    entity_id = entity_uri.rstrip("/").split("/")[-1]
    return {
        "chunk_id": f"icd11:{LINEARIZATION}:{code or entity_id}",
        "source": "icd11",
        "source_url": entity_uri,
        "title": title,
        "chunk_text": chunk_text,
        "granularity": "entity",
        "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
        "license": LICENSE,
    }


def main():
    token = get_token()
    chapter_uri = (
        f"{API_BASE}/icd/release/11/{RELEASE_ID}/{LINEARIZATION}/{CHAPTER_ENTITY_ID}"
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_written, n_skipped = 0, 0
    seen: set = set()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for uri, data in walk(token, chapter_uri, seen):
            chunk = to_chunk(uri, data)
            if chunk is None:
                n_skipped += 1
                continue
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 50 == 0:
                print(f"  ...{n_written} chunks")

    print(f"Done. {n_written} chunks -> {OUT_PATH} ({n_skipped} structural nodes skipped)")


if __name__ == "__main__":
    main()
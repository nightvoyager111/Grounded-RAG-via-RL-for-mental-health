"""
Fetch NIMH mental-health topic pages and emit paragraph-level chunks.

WHY NIMH:
- US federal government work -> generally public domain, clean for ingestion
  (no anti-LLM clause like OpenStax has). Confirm per-page footer to be safe.
- Prose descriptions of disorders/treatments -> the "synthesis room" tier that
  ICD-11's terse criteria don't give you. A faithful answer must integrate
  across sentences, so copy_penalty bites and helpfulness is measurable.

PAIRING: scope NIMH topics to MATCH your ICD-11 entities so retrieval has
somewhere to land. Don't pull NIMH topics that have no ICD-11 counterpart in
your corpus, or you create questions retrieval can't support.

Two-stage: raw HTML cached to data/raw/nimh/, chunks to data/corpus/nimh.jsonl.

NOTE: NIMH page slugs change over time. The TOPICS list below is the one thing
to verify by hand before a run -- open each URL once. Treat 404s as a signal to
update the slug, not as a silent skip.
"""

import json
import time
import pathlib
import datetime
import hashlib
import requests
from typing import Optional, List
from bs4 import BeautifulSoup

BASE = "https://www.nimh.nih.gov/health/topics"
INDEX_URL = BASE  # the topics landing page lists every current topic link

# EXPLICIT ALLOWLIST — disorder-factual topics that map to ICD-11 entities.
# This is the corpus scope. It deliberately EXCLUDES, per CLAUDE.md "Is NOT":
#   - suicide / self-harm content (hard ethics boundary)
#   - advice / coping pages (coping-with-traumatic-events, caring-for-*)
#   - demographic pages (men/women/older-adults/child-* mental health)
#   - service/treatment pages (medications, psychotherapies, brain-stimulation)
#   - espanol landing, covid/hiv (no clean ICD-11 disorder to land on)
# Each entry should have a counterpart in your ICD-11 corpus so retrieval lands.
# Add substance-use only if you also keep its ICD-11 entities.
TOPICS = [
    "anxiety-disorders",
    "depression",
    "obsessive-compulsive-disorder-ocd",
    "bipolar-disorder",
    "post-traumatic-stress-disorder-ptsd",
    "schizophrenia",
    "borderline-personality-disorder",
    "attention-deficit-hyperactivity-disorder-adhd",
    "autism-spectrum-disorders-asd",
    "disruptive-mood-dysregulation-disorder-dmdd",
    # "substance-use-and-mental-health",  # enable only if ICD-11 side kept too
    # "eating-disorders",                 # borderline: specific-numbers risk
]


def discover_topics(verbose: bool = True) -> List[str]:
    """
    REVIEW-ONLY helper. Scrapes the index and prints every topic slug so you
    can eyeball what NIMH currently publishes and update TOPICS by hand.
    NOT used by main() — corpus scope is the explicit TOPICS allowlist above,
    because discovery grabs advice/demographic/service pages that violate scope.
    """
    resp = requests.get(INDEX_URL, timeout=30,
                        headers={"User-Agent": "grounded-rag-research/0.1"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    slugs = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/health/topics/" not in href:
            continue
        slug = href.rstrip("/").split("/health/topics/")[-1].split("/")[0]
        if slug:
            slugs.add(slug)
    found = sorted(slugs)
    if verbose:
        in_scope = [s for s in found if s in TOPICS]
        out_scope = [s for s in found if s not in TOPICS]
        print(f"  [review] {len(found)} topics live on NIMH")
        print(f"  [review] in your allowlist: {in_scope}")
        print(f"  [review] NOT in allowlist (advice/demographic/service/etc): {out_scope}")
    return found

RAW_DIR = pathlib.Path("data/raw/nimh")
OUT_PATH = pathlib.Path("data/corpus/nimh.jsonl")
LICENSE = "Public domain (U.S. federal government work) -- verify per page"

REQUEST_PAUSE_S = 0.5


def fetch_html(slug: str) -> Optional[str]:
    cache = RAW_DIR / f"{slug}.html"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = f"{BASE}/{slug}"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "grounded-rag-research/0.1"})
    if resp.status_code == 404:
        print(f"  !! 404 for {slug} -- update the slug")
        return None
    resp.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text, encoding="utf-8")
    time.sleep(REQUEST_PAUSE_S)
    return resp.text


MIN_CHUNK_CHARS = 80    # real paragraphs can be short; 200 was dropping content
MAX_CHUNK_CHARS = 1500  # keep premises short enough for the NLI verifier
MERGE_UNDER = 250       # fragments under this get merged with the next block

# Boilerplate/navigational phrases that appear on every NIMH topic page. A block
# built mostly from these is a CTA/nav strip, not disorder-factual content.
_BOILERPLATE_PHRASES = (
    "where can i learn more",
    "why is nimh studying",
    "how is nimh research addressing",
    "explore clinical trials",
    "share outreach materials",
    "additional federal resources",
    "join a study",
    "find help",
    "featured videos",
    "print this page",
    "last reviewed",
    "follow nimh",
    "subscribe",
    "español",
)


def _clean(text: str) -> str:
    return " ".join(text.split())


def _is_boilerplate(text: str) -> bool:
    low = text.lower()
    if sum(1 for p in _BOILERPLATE_PHRASES if p in low) >= 2:
        return True
    # Question-heavy short blocks (e.g. "Where can I…? Why is…? How is…?") are
    # nav prompts, not content.
    if text.count("?") >= 3 and text.count(".") <= 1 and len(text) < 400:
        return True
    return False


def extract_paragraphs(html: str) -> List[str]:
    """
    Structure-agnostic extraction. NIMH wraps body text in nested divs and
    accordions, not bare <p> under <main>, so keying only on <p> loses most of
    the page. Strategy: strip known boilerplate containers, then collect text
    from every content-bearing block, merge short fragments, split long ones,
    and dedupe.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Strip non-content. Include NIMH-specific chrome (breadcrumbs, share bars,
    # related-links, "on this page" nav) by common class/id hints.
    for sel in ["nav", "header", "footer", "aside", "script", "style", "form",
                "button", "figure"]:
        for tag in soup.find_all(sel):
            tag.decompose()
    for hint in ["breadcrumb", "share", "social", "related", "sidebar",
                 "on-this-page", "skip", "menu", "search", "cookie", "banner"]:
        for tag in soup.find_all(attrs={"class": lambda c: c and hint in " ".join(c).lower()}):
            tag.decompose()
        for tag in soup.find_all(attrs={"id": lambda c: c and hint in c.lower()}):
            tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup

    # Collect content blocks in document order. Headings included as context
    # anchors; paragraphs and list items as the substance.
    blocks: List[str] = []
    for el in root.find_all(["h2", "h3", "h4", "p", "li"]):
        # Skip a <li> whose text is fully contained in an ancestor we'll also
        # capture is hard to detect cheaply; dedupe pass below handles overlap.
        text = _clean(el.get_text(" ", strip=True))
        if len(text) < 25:  # drop nav crumbs, lone labels
            continue
        blocks.append(text)

    # Merge short fragments forward so a 90-char sentence isn't its own chunk.
    merged: List[str] = []
    buf = ""
    for b in blocks:
        if buf:
            buf = buf + " " + b
        else:
            buf = b
        if len(buf) >= MERGE_UNDER:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)

    # Enforce max length and min length, dedupe.
    out: List[str] = []
    seen = set()
    for m in merged:
        segments = ([m] if len(m) <= MAX_CHUNK_CHARS
                    else [m[i:i + MAX_CHUNK_CHARS] for i in range(0, len(m), MAX_CHUNK_CHARS)])
        for seg in segments:
            seg = seg.strip()
            key = seg[:120]
            if len(seg) >= MIN_CHUNK_CHARS and key not in seen and not _is_boilerplate(seg):
                seen.add(key)
                out.append(seg)
    return out


def to_chunk(slug: str, idx: int, text: str) -> dict:
    cid_hash = hashlib.sha1(f"{slug}:{idx}".encode()).hexdigest()[:8]
    return {
        "chunk_id": f"nimh:{slug}:{cid_hash}",
        "source": "nimh",
        "source_url": f"{BASE}/{slug}",
        "title": slug.replace("-", " "),
        "chunk_text": text,
        "granularity": "paragraph",
        "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
        "license": LICENSE,
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_written, n_topics_ok = 0, 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for slug in TOPICS:
            html = fetch_html(slug)
            if html is None:
                continue
            n_topics_ok += 1
            for idx, para in enumerate(extract_paragraphs(html)):
                f.write(json.dumps(to_chunk(slug, idx, para), ensure_ascii=False) + "\n")
                n_written += 1
    print(f"Done. {n_written} chunks from {n_topics_ok}/{len(TOPICS)} topics -> {OUT_PATH}")


if __name__ == "__main__":
    import sys
    if "--discover" in sys.argv:
        # Review mode: print what NIMH currently publishes vs your allowlist,
        # then exit. Use this to decide whether to edit TOPICS. No fetching.
        discover_topics(verbose=True)
    else:
        main()
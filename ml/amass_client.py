"""
Amass-Anbindung: passende Literatur und klinische Studien zu den
Tumor-Kennzahlen aus tumor_features.compute_features().

API-Doku: https://platform.amass.tech/documentation/for-ai-agents/llm-quick-reference
Key:      AMASS_API_KEY in .env im Projektordner

Smoke-Test:
    .venv/bin/python ml/amass_client.py
"""

from pathlib import Path
import json
import os
import re
import time

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).parent.parent

load_dotenv(ROOT / ".env")

BASE_URL = "https://api.amass.tech/api/v1"

# Letzte erfolgreiche Antwort; Fallback, falls Amass in der Demo
# nicht erreichbar ist (Netzwerk, Rate-Limit, Credits)
CACHE_FILE = ROOT / "outputs" / "demo_literature.json"

TIMEOUT_SECONDS = 20

PAPER_LIMIT = 5
TRIAL_LIMIT = 3


class AmassError(Exception):
    pass


# ============================================================
# HTTP
# ============================================================

def _get(core, params):

    key = os.getenv("AMASS_API_KEY")

    if not key:
        raise AmassError("AMASS_API_KEY is missing in .env")

    url = f"{BASE_URL}/cores/{core}/records"

    headers = {"Authorization": f"Bearer {key}"}

    for attempt in range(2):

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=TIMEOUT_SECONDS
            )
        except requests.RequestException as error:
            raise AmassError(f"Amass not reachable: {error}")

        # Rate-Limit (60 Anfragen/Minute): einmal warten, dann erneut
        if response.status_code == 429 and attempt == 0:
            time.sleep(int(response.headers.get("Retry-After", "2")))
            continue

        if not response.ok:
            try:
                message = response.json()["error"]["message"]
            except Exception:
                message = response.text[:200]
            raise AmassError(f"Amass HTTP {response.status_code}: {message}")

        return response.json()["data"]

    raise AmassError("Amass rate limit exceeded")


# ============================================================
# QUERY
# ============================================================

def build_query_terms(features):
    """
    Übersetzt die Kennzahlen in Standard-Fachbegriffe.
    ET/TC/WT sind nur unsere interne Volumenberechnung und
    tauchen in der Literatur so nicht auf.
    """

    seg = features["segmentation"]
    ratios = features["ratios"]
    lesion_count = features["spatial"]["lesion_count"]

    terms = ["glioma", "MRI"]

    if seg["enhancing_tumor_volume_ml"] > 0:
        terms.append("enhancing tumor")
    else:
        terms.append("non-enhancing")

    # None, wenn kein Tumor segmentiert wurde
    if (ratios.get("edema_fraction") or 0) >= 0.5:
        terms.append("peritumoral edema")

    if lesion_count and lesion_count > 1:
        terms.append("multifocal")

    terms.append("tumor volume")

    return terms


def matched_terms(terms, *texts):
    """
    Plausibilitätscheck: welche Suchbegriffe tauchen im Treffer
    tatsächlich auf? Beantwortet "warum genau dieses Paper?".
    """

    haystack = " ".join(t for t in texts if t).lower()

    return [
        term for term in terms
        if all(word in haystack for word in term.lower().split())
    ]


def shorten(text, max_chars=280):
    """Kürzt auf ganze Sätze, damit Bericht und UI lesbar bleiben."""

    text = re.sub(r"\s+", " ", text or "").strip()

    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]

    sentence_end = cut.rfind(". ")

    if sentence_end > max_chars // 2:
        return cut[:sentence_end + 1]

    return cut.rsplit(" ", 1)[0] + " …"


# ============================================================
# PAPERS + TRIALS
# ============================================================

def get_literature_references(features: dict) -> list[dict]:
    """
    Input:  Output von tumor_features.compute_features()
    Output: [{"title": str, "summary": str, "url": str,
              "journal": str, "year": str, "citations": int,
              "matched_terms": [str, ...]}, ...]
    """

    terms = build_query_terms(features)

    records = _get("biomedcore", {
        "query": " ".join(terms),
        "limit": PAPER_LIMIT * 2,
        "isRetracted": "false",
        "minPublicationDate": "2015-01-01",
    })

    papers = []

    for record in records:

        matches = matched_terms(
            terms,
            record.get("title"),
            record.get("abstract"),
            " ".join(record.get("keywords") or []),
        )

        # mindestens ein Begriff außer den allgemeinen "glioma"/"MRI"
        if not set(matches) - {"glioma", "MRI"}:
            continue

        if record.get("doi"):
            url = f"https://doi.org/{record['doi']}"
        elif record.get("pmid"):
            url = f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/"
        else:
            continue

        papers.append({
            "title": record.get("title", "").strip(),
            "summary": shorten(record.get("abstract")),
            "url": url,
            "journal": record.get("journal") or "",
            "year": (record.get("publicationDate") or "")[:4],
            "citations": record.get("citationCount") or 0,
            "matched_terms": matches,
        })

        if len(papers) == PAPER_LIMIT:
            break

    return papers


def get_clinical_trials(features: dict) -> list[dict]:
    """
    Input:  Output von tumor_features.compute_features()
    Output: [{"title": str, "summary": str, "url": str,
              "trial_id": str, "phase": str, "status": str}, ...]
            nur aktuell rekrutierende Studien
    """

    terms = build_query_terms(features)

    # Studien werden nach Erkrankung registriert, nicht nach
    # einzelnen Bildmerkmalen -> Erkrankung + Bildgebung suchen
    # (nur "glioma" liefert auch fachfremde Studien, z.B. Urin-Marker)
    records = _get("trialcore", {
        "query": "glioma MRI imaging tumor volume",
        "limit": TRIAL_LIMIT,
        "overallStatus": "RECRUITING",
    })

    trials = []

    for record in records:

        trial_id = record.get("nctId") or record.get("registryId") or ""

        url = record.get("sourceUrl") or (
            f"https://clinicaltrials.gov/study/{trial_id}"
            if trial_id.startswith("NCT") else ""
        )

        if not url:
            continue

        phase = record.get("phase") or ""

        trials.append({
            "title": record.get("briefTitle") or record.get("officialTitle") or "",
            "summary": shorten(record.get("briefSummary")),
            "url": url,
            "trial_id": trial_id,
            "phase": "" if phase == "NA" else phase.replace("_", " ").title(),
            "status": record.get("overallStatus") or "",
            "matched_terms": matched_terms(
                terms,
                record.get("briefTitle"),
                record.get("briefSummary"),
            ),
        })

    return trials


def get_research(features: dict) -> dict:
    """
    Papers + Trials zusammen, mit Cache-Fallback.
    Output: {"papers": [...], "trials": [...], "query_terms": [...],
             "source": "live" | "cache"}
    """

    try:

        research = {
            "papers": get_literature_references(features),
            "trials": get_clinical_trials(features),
            "query_terms": build_query_terms(features),
            "source": "live",
        }

        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

        CACHE_FILE.write_text(
            json.dumps(research, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        return research

    except AmassError as error:

        if not CACHE_FILE.exists():
            raise

        print(f"Amass failed ({error}), using cached results.")

        research = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

        research["source"] = "cache"

        return research


# ============================================================
# SMOKE-TEST
# ============================================================

if __name__ == "__main__":

    mock_features = {
        "segmentation": {
            "whole_tumor_volume_ml": 83.9,
            "tumor_core_volume_ml": 3.5,
            "enhancing_tumor_volume_ml": 3.0,
            "edema_volume_ml": 80.5,
        },
        "ratios": {
            "enhancing_fraction": 0.036,
            "core_fraction": 0.041,
            "edema_fraction": 0.959,
        },
        "spatial": {"lesion_count": 1},
    }

    print("Query:", " ".join(build_query_terms(mock_features)))

    research = get_research(mock_features)

    print(f"\nSource: {research['source']}")

    print(f"\n===== PAPERS ({len(research['papers'])}) =====")

    for paper in research["papers"]:
        print(f"\n- {paper['title']} ({paper['journal']}, {paper['year']})")
        print(f"  {paper['url']}")
        print(f"  matched: {', '.join(paper['matched_terms'])}")

    print(f"\n===== TRIALS ({len(research['trials'])}) =====")

    for trial in research["trials"]:
        print(f"\n- {trial['title']} [{trial['trial_id']}, {trial['phase']}]")
        print(f"  {trial['url']}")

"""
Clinical-Decision-Support-Bericht mit einem LLM über Groq.

MONAI sieht das MRI, Amass findet die Evidenz, das LLM liest
Befund + Evidenz und erklärt beides dem Arzt.

Aufteilung, damit der Bericht verlässlich bleibt:
- das LLM schreibt nur die Texte (strukturiertes JSON)
- Zahlen (Volumentabelle) und Quellenliste setzen wir selbst ein
- das LLM darf nur auf die Quellen [1]..[n] verweisen, die wir mitschicken
- das LLM bekommt keine Patienten-ID
- fällt das LLM aus, gibt es den regelbasierten Bericht aus report.py

API-Doku: https://console.groq.com/docs/structured-outputs
Key:  GROQ_API_KEY in .env im Projektordner
Test: .venv/bin/python ml/llm_report.py
"""

from pathlib import Path
from datetime import date
import json
import os
import re

import requests
from dotenv import load_dotenv

from report import (
    DISCLAIMER,
    generate_report,
    safe_text,
    safe_url,
    volumetry_lines,
)


ROOT = Path(__file__).parent.parent

load_dotenv(ROOT / ".env")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# unterstützt strikte JSON-Schemas bei Groq
LLM_MODEL = "openai/gpt-oss-120b"

TIMEOUT_SECONDS = 90

# Strict-Mode kennt kein maxItems -> Listen selbst kürzen
MAX_BULLETS = 5


class LLMError(Exception):
    pass


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """\
You are a clinical decision support assistant for neuroradiology and
neuro-oncology. You write a structured pre-read for a treating physician,
who makes all final decisions.

INPUT
1. CASE DATA: quantitative results of an automated MONAI SegResNet brain
   tumor segmentation of a multiparametric MRI, plus a list of what is NOT
   available for this case.
2. EVIDENCE: a numbered list [1]..[n] of papers and recruiting clinical
   trials retrieved from the Amass literature database by keyword match.

HARD RULES (a report that breaks one of these is unusable)
- Use only CASE DATA and EVIDENCE. Never invent measurements, anatomical
  location or laterality, normative comparisons (e.g. standard deviations,
  percentiles), patient history, symptoms, lab values, or prior imaging.
- Anything listed under NOT AVAILABLE must not be described as if known.
  If it matters for a section, say explicitly that it was not provided.
- Cite evidence only as [n] with numbers from the EVIDENCE list, and only
  for statements the cited title/summary actually supports. If no item
  supports a statement, do not cite anything for it.
- Never state a definitive diagnosis or grade. Differential diagnoses are
  hypotheses "to be considered", ranked by how well they fit the
  quantitative imaging pattern.
- Next steps are suggestions for the physician to consider (e.g. further
  imaging sequences, tissue diagnosis, multidisciplinary tumor board).
  No specific drugs, doses, or surgical plans.

STYLE
- Professional clinical English, concise, no markdown, no bullet symbols.
- Quote volumes in mL with one decimal, exactly as in CASE DATA.
- Each list: 2-5 items, one sentence each.
"""


def _object(properties):
    """Strict-Mode-Objekt: alle Felder required, keine Zusatzfelder."""

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


STRING = {"type": "string"}

STRING_LIST = {"type": "array", "items": STRING}


RESPONSE_SCHEMA = _object({

    # 1. Synthesis & key findings
    "quantitative_findings": {
        **STRING,
        "description": "2-3 sentences: the key quantitative abnormalities.",
    },
    "clinical_correlation": {
        **STRING,
        "description": (
            "1-2 sentences. No clinical history is available: state this and "
            "name which clinical information would be needed to correlate."
        ),
    },

    # 2. Differential diagnosis
    "primary_consideration": _object({
        "condition": STRING,
        "imaging_evidence": {
            **STRING,
            "description": "Which case findings fit this hypothesis.",
        },
        "literature_support": {
            **STRING,
            "description": "One sentence with [n] citations, or 'None of "
                           "the retrieved evidence addresses this directly.'",
        },
    }),
    "alternatives": {
        "type": "array",
        "items": _object({
            "condition": STRING,
            "arguments_against": {
                **STRING,
                "description": "Which case findings argue against it, or "
                               "what would be needed to rule it out.",
            },
        }),
    },

    # 3. Suggested next steps
    "next_steps": STRING_LIST,

    # 4. Limitations
    "limitations": STRING_LIST,
})


def build_evidence(research):
    """Papers + Trials als eine durchnummerierte Liste [1]..[n]."""

    evidence = []

    for paper in (research or {}).get("papers") or []:
        evidence.append({**paper, "kind": "paper"})

    for trial in (research or {}).get("trials") or []:
        evidence.append({**trial, "kind": "trial"})

    return evidence


def build_user_message(features, evidence):

    findings = {
        "segmentation_ml": features["segmentation"],
        "fractions_of_whole_tumor": features["ratios"],
        "lesion_count": features["spatial"]["lesion_count"],
        "model": features["model"],
        "mri_sequences_used": ["T1", "T1ce (contrast-enhanced)", "T2", "FLAIR"],
        "label_definitions": {
            "whole_tumor": "all abnormal tissue incl. peritumoral edema",
            "tumor_core": "necrotic/non-enhancing core + enhancing tumor",
            "enhancing_tumor": "contrast-enhancing tissue on T1ce",
            "edema": "peritumoral T2/FLAIR hyperintensity outside the core",
        },
        "not_available": [
            "anatomical location and laterality of the lesion",
            "normative or population reference values",
            "patient age, sex, symptoms, and clinical history",
            "prior imaging or treatment history",
            "DWI/ADC, perfusion, spectroscopy",
            "histopathology and molecular markers",
            "radiologist review of the segmentation",
        ],
    }

    evidence_text = []

    for i, item in enumerate(evidence, start=1):

        if item["kind"] == "paper":
            meta = f"Paper, {item.get('journal', '')} {item.get('year', '')}"
        else:
            meta = (
                f"Recruiting clinical trial {item.get('trial_id', '')} "
                f"{item.get('phase', '')}"
            )

        evidence_text.append(
            f"[{i}] {item['title']} ({meta.strip()})\n"
            f"    {item.get('summary', '')}"
        )

    return (
        "CASE DATA (JSON):\n"
        f"{json.dumps(findings, indent=2)}\n\n"
        "EVIDENCE:\n"
        + ("\n".join(evidence_text) if evidence_text else "(none available)")
    )


# ============================================================
# LLM (Groq)
# ============================================================

def ask_llm(features, evidence):

    key = os.getenv("GROQ_API_KEY")

    if not key:
        raise LLMError("GROQ_API_KEY is missing in .env")

    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(features, evidence)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "clinical_report",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            },
        },
    }

    try:
        response = requests.post(
            GROQ_URL,
            json=body,
            headers={"Authorization": f"Bearer {key}"},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise LLMError(f"Groq not reachable: {error}")

    if not response.ok:
        raise LLMError(f"Groq HTTP {response.status_code}: {response.text[:300]}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, ValueError) as error:
        raise LLMError(f"Unexpected Groq response: {error}")


# ============================================================
# RENDERING
# ============================================================

def clean_citations(text, evidence_count):
    """
    Verweise auf nicht existierende Quellen entfernen,
    auch in Listen wie [1,3] oder [2, 9].
    """

    def keep(match):
        numbers = [
            n for n in re.findall(r"\d+", match.group(0))
            if 1 <= int(n) <= evidence_count
        ]
        return f" [{', '.join(numbers)}]" if numbers else ""

    return re.sub(r"\s?\[\d+(?:\s*[,–-]\s*\d+)*\]", keep, text)


def evidence_meta(item):
    """z.B. 'PloS one, 2017' oder 'Recruiting trial, NCT03684109'."""

    if item["kind"] == "paper":
        parts = [item.get("journal"), item.get("year")]
    else:
        parts = ["Recruiting trial", item.get("trial_id"), item.get("phase")]

    return ", ".join(part for part in parts if part)


def render_markdown(features, answer, evidence):

    n = len(evidence)

    def text(value):
        return safe_text(clean_citations(value, n))

    def bullets(items):
        return [f"- {text(item)}" for item in items[:MAX_BULLETS]] or ["- None"]

    primary = answer["primary_consideration"]

    lines = [
        "# Clinical Decision Support",
        "",
        f"**Patient:** {safe_text(features['patient_id'])}  ",
        f"**Date:** {date.today().strftime('%d %b %Y')}  ",
        f"**Method:** {features['analysis_type']} ({features['model']}), "
        f"evidence via Amass, summary by {LLM_MODEL}",
        "",
        f"> ⚠️ {DISCLAIMER}",
        "",
        "## 1. Synthesis & Key Findings",
        "",
        f"- **Quantitative findings:** {text(answer['quantitative_findings'])}",
        f"- **Clinical correlation:** {text(answer['clinical_correlation'])}",
        "",
    ]

    lines += volumetry_lines(features)

    lines += [
        "---",
        "",
        "## 2. Differential Diagnosis",
        "",
        # Unterpunkte mit 4 Leerzeichen, Python-Markdown braucht 4 statt 2
        f"- **Primary consideration:** {text(primary['condition'])}",
        f"    - *Imaging evidence:* {text(primary['imaging_evidence'])}",
        f"    - *Literature support:* {text(primary['literature_support'])}",
        "- **Alternatives to consider:**",
    ]

    for alternative in answer["alternatives"][:MAX_BULLETS] or []:
        lines.append(
            f"    - {text(alternative['condition'])} – "
            f"*Argues against:* {text(alternative['arguments_against'])}"
        )

    if not answer["alternatives"]:
        lines.append("    - None suggested")

    lines += [
        "",
        "---",
        "",
        "## 3. Suggested Next Steps",
        "",
        *bullets(answer["next_steps"]),
        "",
        "*Suggestions only; all decisions remain with the treating "
        "physician.*",
        "",
        "---",
        "",
        "## 4. Limitations of the Automated Analysis",
        "",
        *bullets(answer["limitations"]),
        "",
    ]

    if evidence:

        lines += ["---", "", "## 5. Evidence", ""]

        for i, item in enumerate(evidence, start=1):

            url = safe_url(item["url"])

            title = safe_text(item["title"])

            link = f"[{title}]({url})" if url else title

            lines.append(
                f"\\[{i}\\] {link} – *{safe_text(evidence_meta(item))}*  "
            )

        lines.append("")

    lines += [
        f"*Text generated by {LLM_MODEL} from the segmentation findings and "
        "the listed evidence only. Evidence was selected automatically and "
        "has not been reviewed by a clinician.*",
        "",
    ]

    return "\n".join(lines)


# ============================================================
# ÖFFENTLICHE FUNKTION
# ============================================================

def build_overview(answer, evidence_count):
    """
    Kurzfassung für den Overview-Tab im Frontend.
    Reiner Text (das Frontend escaped selbst), Verweise bereinigt.
    """

    def text(value):
        return clean_citations(value, evidence_count).strip()

    primary = answer["primary_consideration"]

    return {
        "summary": text(answer["quantitative_findings"]),
        "primary_consideration": text(primary["condition"]),
        "primary_evidence": text(primary["imaging_evidence"]),
        "next_steps": [text(step) for step in answer["next_steps"][:MAX_BULLETS]],
    }


def evidence_list(evidence):
    """Quellenliste für den Evidence-Tab im Frontend."""

    return [
        {
            "number": i,
            "kind": item["kind"],
            "title": item["title"],
            "url": safe_url(item["url"]),
            "meta": evidence_meta(item),
            "summary": item.get("summary", ""),
        }
        for i, item in enumerate(evidence, start=1)
    ]


def generate_clinical_report(
    features: dict,
    research: dict | None = None
) -> dict:
    """
    Input:  features – Output von tumor_features.compute_features()
            research – Output von amass_client.get_research() oder None
    Output: {
        "markdown": str,             # vollständiger Bericht
        "source": "llm" | "fallback",
        "overview": {...} | None,    # None beim regelbasierten Fallback
        "evidence": [{"number", "kind", "title", "url", "meta", "summary"}]
    }
    """

    evidence = build_evidence(research)

    try:
        answer = ask_llm(features, evidence)
    except LLMError as error:
        print(f"LLM report failed ({error}), using rule-based report.")
        return {
            "markdown": generate_report(features, research),
            "source": "fallback",
            "overview": None,
            "evidence": evidence_list(evidence),
        }

    return {
        "markdown": render_markdown(features, answer, evidence),
        "source": "llm",
        "overview": build_overview(answer, len(evidence)),
        "evidence": evidence_list(evidence),
    }


# ============================================================
# LOKALER TEST
# ============================================================

if __name__ == "__main__":

    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    from amass_client import get_research

    mock_features = {
        "patient_id": "BRATS_457",
        "analysis_type": "automated MRI segmentation",
        "model": "MONAI SegResNet",
        "segmentation": {
            "whole_tumor_volume_ml": 83.94,
            "tumor_core_volume_ml": 3.47,
            "enhancing_tumor_volume_ml": 2.98,
            "edema_volume_ml": 80.48,
        },
        "ratios": {
            "enhancing_fraction": 0.036,
            "core_fraction": 0.041,
            "edema_fraction": 0.959,
        },
        "spatial": {"lesion_count": 1},
    }

    report = generate_clinical_report(
        mock_features,
        get_research(mock_features)
    )

    print(f"Source: {report['source']}\n")
    print(json.dumps(report["overview"], indent=2))
    print(report["markdown"])

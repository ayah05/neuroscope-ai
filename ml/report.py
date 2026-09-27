from pathlib import Path
from datetime import date
import html
import json
import re


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).parent.parent

FEATURES_FILE = ROOT / "outputs" / "brats_457_features.json"
LITERATURE_FILE = ROOT / "outputs" / "demo_literature.json"
OUTPUT_FILE = ROOT / "outputs" / "brats_457_report.md"


DISCLAIMER = (
    "This report was generated automatically from an AI segmentation "
    "and is intended for research and demonstration purposes only. "
    "It does not replace a radiological assessment and must not be used "
    "for diagnostic or treatment decisions."
)


# ============================================================
# HELPERS
# ============================================================

def ml(value):

    return f"{value:.1f} mL"


def percent(value):

    # tumor_features liefert None, wenn kein Tumor segmentiert wurde
    if value is None:
        return "n/a"

    return f"{value * 100:.0f}%"


def method_label(features):
    """
    'Automated MRI segmentation (MONAI SegResNet)'.
    features["model"] ist ein Dict ({"name": ...}) oder ein String.
    """

    model = features.get("model")

    name = model.get("name") if isinstance(model, dict) else model

    return f"Automated MRI segmentation ({name or 'MONAI SegResNet'})"


def safe_text(value):
    """
    Externe Texte (Amass) für Markdown -> HTML entschärfen:
    kein eingeschleustes HTML, keine kaputten Markdown-Links.
    """

    text = html.escape(str(value or ""))

    return re.sub(r"([\[\]*_`])", r"\\\1", text)


def safe_url(value):
    """Nur http(s)-Links; Zeichen, die Markdown-Links brechen, kodieren."""

    url = str(value or "")

    if not re.match(r"^https?://", url):
        return None

    return (
        url.replace(" ", "%20")
        .replace("(", "%28")
        .replace(")", "%29")
        .replace("<", "%3C")
        .replace(">", "%3E")
        .replace('"', "%22")
    )


def reference_lines(index, item, details):

    url = safe_url(item["url"])

    title = safe_text(item["title"])

    heading = f"**[{title}]({url})**" if url else f"**{title}**"

    lines = [f"{index}. {heading}  "]

    if details:
        lines.append(f"   *{safe_text(details)}*  ")

    if item.get("summary"):
        lines.append(f"   {safe_text(item['summary'])}  ")

    if item.get("matched_terms"):
        lines.append(
            f"   Matched terms: {safe_text(', '.join(item['matched_terms']))}"
        )

    return lines


def describe_findings(features):
    """
    Beschreibt die Messwerte in ganzen Sätzen.
    Bewusst rein deskriptiv: keine Diagnose, keine Graduierung.
    """

    seg = features["segmentation"]
    ratios = features["ratios"]
    lesion_count = features["spatial"]["lesion_count"]

    if seg["whole_tumor_volume_ml"] == 0:
        return ["No tumor tissue was detected in the segmentation."]

    sentences = []

    # Anzahl Läsionen
    if lesion_count == 1:
        sentences.append(
            "A single contiguous (solitary) lesion was segmented."
        )
    else:
        sentences.append(
            f"{lesion_count} spatially separate lesions were segmented "
            f"(multifocal pattern)."
        )

    # Gesamtausdehnung
    sentences.append(
        f"The whole tumor volume, including peritumoral edema, is "
        f"{ml(seg['whole_tumor_volume_ml'])}."
    )

    # Ödem
    sentences.append(
        f"Of this, {ml(seg['edema_volume_ml'])} "
        f"({percent(ratios['edema_fraction'])}) corresponds to "
        f"peritumoral edema."
    )

    # Kontrastmittelanreicherung
    if seg["enhancing_tumor_volume_ml"] > 0:
        sentences.append(
            f"A contrast-enhancing component is present "
            f"({ml(seg['enhancing_tumor_volume_ml'])}, "
            f"{percent(ratios['enhancing_fraction'])} of the whole tumor)."
        )
    else:
        sentences.append(
            "No contrast-enhancing component was segmented."
        )

    return sentences


def volumetry_lines(features):
    """Volumentabelle direkt aus den Messwerten (auch von llm_report.py genutzt)."""

    seg = features["segmentation"]
    ratios = features["ratios"]

    return [
        "| Region | Volume | Share of whole tumor |",
        "|---|---|---|",
        f"| Whole tumor (WT) | {ml(seg['whole_tumor_volume_ml'])} | 100% |",
        f"| Tumor core (TC) | {ml(seg['tumor_core_volume_ml'])} "
        f"| {percent(ratios['core_fraction'])} |",
        f"| Enhancing tumor (ET) "
        f"| {ml(seg['enhancing_tumor_volume_ml'])} "
        f"| {percent(ratios['enhancing_fraction'])} |",
        f"| Peritumoral edema | {ml(seg['edema_volume_ml'])} "
        f"| {percent(ratios['edema_fraction'])} |",
        "",
        "*WT = tumor core + edema; TC = necrotic/non-enhancing "
        "component + enhancing tumor.*",
        "",
    ]


# ============================================================
# REPORT
# ============================================================

def generate_report(
    features: dict,
    research: dict | None = None
) -> str:
    """
    Input:  features – Output von tumor_features.compute_features()
            research – optional, Output von amass_client.get_research():
                       {"papers": [...], "trials": [...], "source": ...}
    Output: Befundbericht (Englisch) als Markdown-String
    """

    lines = [
        "# Automated MRI Report – Brain Tumor Segmentation",
        "",
        f"**Patient:** {features['patient_id']}  ",
        f"**Date:** {date.today().strftime('%d %b %Y')}  ",
        f"**Method:** {method_label(features)}  ",
        "**Sequences:** T1, T1ce, T2, FLAIR",
        "",
        f"> ⚠️ {DISCLAIMER}",
        "",
        "## 1. Findings",
        "",
    ]

    lines += describe_findings(features)

    lines += [
        "",
        "## 2. Volumetry",
        "",
    ]

    lines += volumetry_lines(features)

    papers = (research or {}).get("papers") or []
    trials = (research or {}).get("trials") or []

    if papers:

        lines += [
            "## 3. Relevant Literature",
            "",
        ]

        for i, paper in enumerate(papers, start=1):

            details = ", ".join(
                part for part in [
                    paper.get("journal"),
                    paper.get("year"),
                    f"{paper['citations']} citations"
                    if paper.get("citations") else ""
                ] if part
            )

            lines += reference_lines(i, paper, details)

        lines.append("")

    if trials:

        lines += [
            f"## {4 if papers else 3}. Recruiting Clinical Trials",
            "",
        ]

        for i, trial in enumerate(trials, start=1):

            details = ", ".join(
                part for part in [trial.get("trial_id"), trial.get("phase")]
                if part
            )

            lines += reference_lines(i, trial, details)

        lines.append("")

    if papers or trials:

        source = (
            "cached results (Amass was not reachable)"
            if research.get("source") == "cache"
            else "live search"
        )

        lines += [
            f"*References retrieved via the Amass API ({source}). "
            "They are selected by keyword match to the segmentation "
            "findings and have not been reviewed by a clinician.*",
            "",
        ]

    return "\n".join(lines)


# ============================================================
# LOKALER TEST
# ============================================================

if __name__ == "__main__":

    with open(FEATURES_FILE, encoding="utf-8") as file:
        features = json.load(file)

    research = None

    # Cache von amass_client.get_research()
    if LITERATURE_FILE.exists():
        with open(LITERATURE_FILE, encoding="utf-8") as file:
            research = json.load(file)

    report = generate_report(features, research)

    print(report)

    OUTPUT_FILE.write_text(report, encoding="utf-8")

    print("\nReport saved to:")
    print(OUTPUT_FILE)

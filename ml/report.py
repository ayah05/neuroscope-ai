from pathlib import Path
from datetime import date
import json


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

    return f"{value * 100:.0f}%"


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


# ============================================================
# REPORT
# ============================================================

def generate_report(
    features: dict,
    literature: list[dict] | None = None
) -> str:
    """
    Input:  features   – Output von tumor_features.compute_features()
            literature – optional, Output von get_literature_references():
                         [{"title": str, "summary": str, "url": str}, ...]
    Output: Befundbericht (Englisch) als Markdown-String
            (direkt nutzbar mit st.markdown() in Streamlit)
    """

    seg = features["segmentation"]
    ratios = features["ratios"]

    lines = [
        "# Automated MRI Report – Brain Tumor Segmentation",
        "",
        f"**Patient:** {features['patient_id']}  ",
        f"**Date:** {date.today().strftime('%d %b %Y')}  ",
        f"**Method:** {features['analysis_type']} ({features['model']})  ",
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

    if literature:

        lines += [
            "## 3. Relevant Literature",
            "",
        ]

        for i, ref in enumerate(literature, start=1):
            lines += [
                f"{i}. **[{ref['title']}]({ref['url']})**  ",
                f"   {ref['summary']}",
            ]

        lines.append("")

    return "\n".join(lines)


# ============================================================
# LOKALER TEST
# ============================================================

if __name__ == "__main__":

    with open(FEATURES_FILE, encoding="utf-8") as file:
        features = json.load(file)

    literature = None

    if LITERATURE_FILE.exists():
        with open(LITERATURE_FILE, encoding="utf-8") as file:
            literature = json.load(file)

    report = generate_report(features, literature)

    print(report)

    OUTPUT_FILE.write_text(report, encoding="utf-8")

    print("\nReport saved to:")
    print(OUTPUT_FILE)

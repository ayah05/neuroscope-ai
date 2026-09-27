"""
NeuroScope Backend

Nimmt einen MRI-Scan aus dem Frontend an, lässt ihn auf Modal segmentieren
und gibt Tumor-Kennzahlen, Schicht-Bild und Befundbericht zurück.

Start (im Projektordner):
    .venv/bin/uvicorn backend.app:app --port 8000
Voraussetzung: Modal-App ist deployed
    .venv/bin/modal deploy ml/modal_inference.py
"""

from pathlib import Path
import base64
import gzip
import io
import json
import re
import sys
import tempfile

import markdown
import matplotlib
import modal
import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from matplotlib.colors import to_rgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(ROOT / "ml"))

from tumor_features import compute_features  # noqa: E402
from llm_report import generate_clinical_report  # noqa: E402
from amass_client import AmassError, get_research  # noqa: E402


MODAL_APP = "neuroscope-monai"
MODAL_FUNCTION = "segment"

# MSD-Kanalreihenfolge im hochgeladenen 4D-Scan
FLAIR_CHANNEL = 0

# Farben der Segmentierung (Okabe-Ito, auch bei Farbenblindheit
# unterscheidbar). Einzige Quelle für Bild UND Legende im Frontend.
SEGMENTATION_CLASSES = [
    {
        "label": 2,
        "name": "Edema",
        "description": "Peritumoral swelling (bright on T2/FLAIR)",
        "color": "#56B4E9",
        "volume_key": "edema_volume_ml",
    },
    {
        "label": 1,
        "name": "Non-enhancing tumor / necrosis",
        "description": "Tumor core without contrast uptake",
        "color": "#F0E442",
        "volume_key": None,  # = Tumorkern - enhancing, siehe legend()
    },
    {
        "label": 4,
        "name": "Enhancing tumor",
        "description": "Contrast-enhancing tissue (bright on T1ce)",
        "color": "#D55E00",
        "volume_key": "enhancing_tumor_volume_ml",
    },
]

OVERLAY_ALPHA = 0.6


app = FastAPI(title="NeuroScope API")


@app.middleware("http")
async def no_stale_frontend(request, call_next):
    """
    Browser sollen Frontend-Dateien vor jeder Nutzung neu prüfen (ETag),
    sonst läuft nach einem Update altes JavaScript gegen das neue Backend.
    """

    response = await call_next(request)

    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"

    return response


# ============================================================
# HELPERS
# ============================================================

NIFTI_SUFFIX = re.compile(r"\.nii(\.gz)?$", flags=re.IGNORECASE)

# Reihenfolge wichtig: t1ce vor t1 prüfen, sonst wird t1ce als t1 erkannt
SEQUENCE_PATTERNS = [
    ("t1ce", re.compile(r"t1[-_]?(ce|c|gd|post|contrast)", re.IGNORECASE)),
    ("flair", re.compile(r"flair", re.IGNORECASE)),
    ("t2", re.compile(r"t2", re.IGNORECASE)),
    ("t1", re.compile(r"t1", re.IGNORECASE)),
]

# MSD-Kanalreihenfolge, die segment() auf Modal erwartet
MSD_ORDER = ["flair", "t1", "t1ce", "t2"]


def safe_id(text):
    """Nur sichere Zeichen für die Patienten-ID."""

    return re.sub(r"[^A-Za-z0-9_-]", "", text).strip("_-") or "MRI scan"


def patient_id_from_filenames(filenames):
    """
    BRATS_457.nii.gz                        -> BRATS_457
    patient_flair.nii.gz, patient_t1.nii.gz -> patient (gemeinsamer Anfang)
    """

    stems = [NIFTI_SUFFIX.sub("", name) for name in filenames]

    if len(stems) == 1:
        return safe_id(stems[0])

    prefix = stems[0]

    for stem in stems[1:]:
        while not stem.startswith(prefix):
            prefix = prefix[:-1]

    return safe_id(prefix)


def detect_sequence(filename):

    stem = NIFTI_SUFFIX.sub("", filename)

    for name, pattern in SEQUENCE_PATTERNS:
        if pattern.search(stem):
            return name

    return None


def spatial_unit(header):
    """
    Räumliche Einheit übernehmen; tumor_features.py rechnet nur mit
    bekannter Einheit. BraTS/MSD-Daten sind in mm.
    """

    unit, _ = header.get_xyzt_units()

    return unit if unit != "unknown" else "mm"


def save_uploaded_sequences(uploads, tmp):
    """
    Vier hochgeladene 3D-Sequenzen speichern.
    Output: {"flair": Path, "t1": Path, "t1ce": Path, "t2": Path}
    """

    paths = {}

    for upload in uploads:

        name = upload.filename or ""

        sequence = detect_sequence(name)

        if sequence is None:
            raise HTTPException(
                400,
                f"Could not tell which sequence '{name}' is. File names must "
                f"contain flair, t1, t1ce or t2."
            )

        if sequence in paths:
            raise HTTPException(
                400,
                f"Two files were detected as {sequence.upper()}. Please "
                f"select exactly one file each for FLAIR, T1, T1ce and T2."
            )

        path = Path(tmp) / f"upload_{sequence}{NIFTI_SUFFIX.search(name).group(0)}"
        path.write_bytes(upload.file.read())

        paths[sequence] = path

    return paths


def combine_sequences(paths, combined_path):
    """
    Vier 3D-Sequenzen -> ein 4D-Scan im MSD-Format [H, W, D, 4]
    mit Kanälen FLAIR, T1, T1ce, T2 (Reihenfolge, die segment() erwartet).
    """

    volumes = {}

    reference = None

    for sequence in MSD_ORDER:

        path = paths[sequence]

        try:
            img = nib.load(path)
            data = np.asarray(img.dataobj, dtype=np.float32)
        except Exception:
            raise HTTPException(
                400,
                f"The {sequence.upper()} file is not a readable NIfTI scan."
            )

        if data.ndim != 3:
            raise HTTPException(
                400,
                f"The {sequence.upper()} file should be a single 3D sequence, "
                f"got shape {data.shape}."
            )

        if reference is not None and data.shape != reference.shape:
            raise HTTPException(
                400,
                f"The sequences have different sizes ({reference.shape} vs "
                f"{data.shape}); they must come from the same scan."
            )

        reference = reference if reference is not None else img

        volumes[sequence] = data

    combined = nib.Nifti1Image(
        np.stack([volumes[s] for s in MSD_ORDER], axis=-1),
        reference.affine
    )

    combined.header.set_xyzt_units(spatial_unit(reference.header))

    nib.save(combined, combined_path)

    return combined_path


def write_sequence_files(scan, mri_dir):
    """
    4D-Scan in die vier Einzeldateien zerlegen, die
    tumor_features.process_patient() im MRI-Ordner erwartet.
    """

    mri_dir.mkdir(parents=True, exist_ok=True)

    data = np.asarray(scan.dataobj, dtype=np.float32)

    for channel, sequence in enumerate(MSD_ORDER):

        img = nib.Nifti1Image(data[..., channel], scan.affine)

        img.header.set_xyzt_units(spatial_unit(scan.header))

        nib.save(img, mri_dir / f"{sequence}.nii.gz")


def load_scan(path):
    """Prüft vor dem GPU-Aufruf, ob der Scan zum Modell passt."""

    try:
        scan = nib.load(path)
        shape = scan.shape
    except Exception:
        raise HTTPException(400, "The file is not a readable NIfTI scan.")

    if len(shape) == 3:
        raise HTTPException(
            400,
            "This file contains only one MRI sequence. Select all 4 sequence "
            "files (FLAIR, T1, T1ce, T2) together, or one combined 4D scan."
        )

    if len(shape) != 4 or shape[3] != 4:
        raise HTTPException(
            400,
            f"Expected a 4D scan with 4 sequences (FLAIR, T1, T1ce, T2), "
            f"got shape {shape}."
        )

    return scan


def load_research(features):
    """
    Amass-Literatur + Studien. Ein Amass-Ausfall (ohne Cache) soll die
    Analyse nicht abbrechen -> dann einfach ohne Literatur.
    """

    try:
        return get_research(features)
    except AmassError as error:
        print(f"Amass unavailable, continuing without research: {error}")
        return None


def research_card(research):
    """Format für die Research-Evidence-Karte im Frontend (results.js)."""

    if research is None:
        return None

    papers = len(research["papers"])
    trials = len(research["trials"])

    summary = (
        f"{papers} peer-reviewed papers and {trials} recruiting clinical "
        f"trials matched the findings ({', '.join(research['query_terms'])}). "
        f"See the report for details."
    )

    if research.get("source") == "cache":
        summary += " Amass was not reachable; showing cached results."

    return {"summary": summary, "papers": papers, "trials": trials}


def legend(features):
    """Legende fürs Frontend: Farbe, Name, Beschreibung, Volumen."""

    seg = features["segmentation"]

    entries = []

    for cls in SEGMENTATION_CLASSES:

        if cls["volume_key"]:
            volume = seg[cls["volume_key"]]
        else:
            volume = round(
                seg["tumor_core_volume_ml"] - seg["enhancing_tumor_volume_ml"],
                2
            )

        entries.append({
            "name": cls["name"],
            "description": cls["description"],
            "color": cls["color"],
            "volume_ml": volume,
        })

    return entries


def render_overlay(flair, segmentation):
    """
    Axiale Schicht mit dem meisten Tumor: FLAIR + farbige Segmentierung.
    Output: (PNG als data-URL, Schichtnummer)
    """

    tumor_per_slice = np.sum(segmentation > 0, axis=(0, 1))

    slice_idx = int(np.argmax(tumor_per_slice))

    background = flair[:, :, slice_idx].T

    mask = segmentation[:, :, slice_idx].T

    # Label -> RGBA, feste Farben aus SEGMENTATION_CLASSES
    overlay = np.zeros(mask.shape + (4,))

    for cls in SEGMENTATION_CLASSES:
        rgb = to_rgb(cls["color"])
        overlay[mask == cls["label"]] = (*rgb, OVERLAY_ALPHA)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

    ax.imshow(background, cmap="gray", origin="lower")

    ax.imshow(overlay, origin="lower", interpolation="nearest")

    ax.axis("off")

    fig.patch.set_facecolor("black")

    buffer = io.BytesIO()

    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)

    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/png;base64,{encoded}", slice_idx


# ============================================================
# API
# ============================================================

@app.post("/api/analyze")
def analyze(images: list[UploadFile] = File(...)):
    """
    Input:  multipart-Feld "images" mit entweder
            - einem 4D-NIfTI (FLAIR, T1, T1ce, T2 in einer Datei), oder
            - vier 3D-NIfTIs, eins pro Sequenz (Name enthält flair/t1/t1ce/t2)
    Output: {
        "patient_id": "BRATS_457",
        "segmentation": {"whole_tumor_ml": 73.8, "tumor_core_ml": 3.3,
                         "enhancing_tumor_ml": 3.2, "edema_ml": 70.5},
        "features": {...},        # kompletter Output von compute_features()
        "report_html": "<h1>...", # Befundbericht
        "report_source": "llm",   # "fallback", wenn das LLM nicht erreichbar
        "overview": {             # Kurzfassung für den Overview-Tab,
            "summary": "...",     # None beim Fallback
            "primary_consideration": "...",
            "primary_evidence": "...",
            "next_steps": ["..."]
        },
        "evidence": [{"number": 1, "kind": "paper", "title": "...",
                      "url": "https://...", "meta": "...", "summary": "..."}],
        "overlay_image": "data:image/png;base64,...",
        "overlay_slice": 77,
        "legend": [{"name": "Edema", "description": "...",
                    "color": "#56B4E9", "volume_ml": 70.53}, ...],
        "research": {"summary": "...", "papers": 5, "trials": 3}
                                  # None, wenn Amass nicht erreichbar
    }
    """

    filenames = [upload.filename or "" for upload in images]

    if not all(NIFTI_SUFFIX.search(name) for name in filenames):
        raise HTTPException(400, "Please upload NIfTI files (.nii or .nii.gz).")

    if len(images) not in (1, 4):
        raise HTTPException(
            400,
            f"Please upload either one 4D scan or the 4 sequence files "
            f"(FLAIR, T1, T1ce, T2) – got {len(images)} files."
        )

    patient_id = patient_id_from_filenames(filenames)

    with tempfile.TemporaryDirectory() as tmp:

        if len(images) == 4:

            scan_path = combine_sequences(
                save_uploaded_sequences(images, tmp),
                Path(tmp) / "scan.nii.gz"
            )

        else:

            # Endung beibehalten: .nii ist nicht gzip-komprimiert
            suffix = NIFTI_SUFFIX.search(filenames[0]).group(0).lower()

            scan_path = Path(tmp) / f"scan{suffix}"
            scan_path.write_bytes(images[0].file.read())

        return run_analysis(scan_path, patient_id, Path(tmp))


def run_analysis(scan_path, patient_id, work_dir):
    """
    Gemeinsame Pipeline für Upload und gespeicherte Patienten:
    4D-Scan -> Modal-Segmentierung -> Kennzahlen -> Amass -> LLM-Bericht.
    Output: Antwort-Dict (Format siehe analyze()).
    """

    scan = load_scan(scan_path)

    scan_bytes = scan_path.read_bytes()

    # segment() auf Modal liest immer .nii.gz
    if scan_path.suffix.lower() == ".nii":
        scan_bytes = gzip.compress(scan_bytes)

    # tumor_features.process_patient() erwartet die Einzelsequenzen
    mri_dir = work_dir / "mri"

    write_sequence_files(scan, mri_dir)

    # ------------------------------------------------------
    # 1. SEGMENTIERUNG AUF MODAL
    # ------------------------------------------------------

    try:
        segment = modal.Function.from_name(MODAL_APP, MODAL_FUNCTION)
        prediction_bytes = segment.remote(scan_bytes)
    except Exception as error:
        raise HTTPException(
            502,
            f"MONAI segmentation on Modal failed: {error}"
        )

    prediction_path = work_dir / "prediction.nii.gz"
    prediction_path.write_bytes(prediction_bytes)

    # ------------------------------------------------------
    # 2. KENNZAHLEN + LITERATUR (Amass) + BERICHT (LLM über Groq)
    # ------------------------------------------------------

    try:
        features = compute_features(prediction_path, patient_id, mri_dir)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(500, f"Feature extraction failed: {error}")

    research = load_research(features)

    report = generate_clinical_report(features, research)

    # ------------------------------------------------------
    # 3. BILD FÜR DEN VIEWER
    # ------------------------------------------------------

    flair = scan.dataobj[..., FLAIR_CHANNEL]
    segmentation = np.asarray(nib.load(prediction_path).dataobj)

    overlay_image, overlay_slice = render_overlay(
        np.asarray(flair),
        segmentation
    )

    seg = features["segmentation"]

    return {
        "patient_id": patient_id,
        "segmentation": {
            "whole_tumor_ml": seg["whole_tumor_volume_ml"],
            "tumor_core_ml": seg["tumor_core_volume_ml"],
            "enhancing_tumor_ml": seg["enhancing_tumor_volume_ml"],
            "edema_ml": seg["edema_volume_ml"]
        },
        "features": features,
        "report_html": markdown.markdown(
            report["markdown"],
            extensions=["tables"]
        ),
        "report_source": report["source"],
        "overview": report["overview"],
        "evidence": report["evidence"],
        "overlay_image": overlay_image,
        "overlay_slice": overlay_slice,
        "legend": legend(features),
        "research": research_card(research)
    }


# ============================================================
# PATIENTS (ml/data/real_patients/<ID>/)
#
# Pro Patient ein Ordner mit flair/t1/t1ce/t2.nii.gz
# (+ optional ground_truth.nii.gz). Optional liegt dort eine
# patient.json mit der Krankengeschichte; Felder wie in
# frontend/js/patients.js dokumentiert (name, birth_date, sex,
# chief_complaint, symptoms, conditions, medications, ...).
# ============================================================

PATIENTS_DIR = ROOT / "ml" / "data" / "real_patients"

PATIENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")

HISTORY_FIELDS = [
    "name", "birth_date", "sex", "status", "last_visit",
    "chief_complaint", "symptoms", "conditions", "medications",
    "allergies", "prior_imaging", "prior_treatment",
    "family_history", "notes",
]


def patient_dir(patient_id):
    """Ordner eines Patienten; schützt vor Pfaden wie '../'."""

    if not PATIENT_ID.match(patient_id):
        raise HTTPException(404, "Patient not found.")

    directory = PATIENTS_DIR / patient_id

    if not directory.is_dir():
        raise HTTPException(404, "Patient not found.")

    return directory


def patient_record(directory):
    """Patient im Format von frontend/js/patients.js."""

    scans = {
        sequence: (directory / f"{sequence}.nii.gz").exists()
        for sequence in MSD_ORDER
    }

    has_scans = all(scans.values())

    record = {
        "id": directory.name,
        "name": directory.name,
        "birth_date": "",
        "sex": "",
        "status": "Scans available" if has_scans else "Incomplete scans",
        "last_visit": "",
        "chief_complaint": "",
        "symptoms": [],
        "conditions": [],
        "medications": [],
        "allergies": [],
        "prior_imaging": [],
        "prior_treatment": [],
        "family_history": "",
        "notes": "",
        "scans": scans,
        "has_scans": has_scans,
        "has_ground_truth": (directory / "ground_truth.nii.gz").exists(),
    }

    history_file = directory / "patient.json"

    if history_file.exists():

        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except ValueError:
            history = {}

        record.update({
            field: history[field]
            for field in HISTORY_FIELDS
            if field in history
        })

    return record


@app.get("/api/patients")
def list_patients():

    if not PATIENTS_DIR.is_dir():
        return []

    return [
        patient_record(directory)
        for directory in sorted(PATIENTS_DIR.iterdir())
        if directory.is_dir() and PATIENT_ID.match(directory.name)
    ]


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str):

    return patient_record(patient_dir(patient_id))


@app.post("/api/patients/{patient_id}/analyze")
def analyze_patient(patient_id: str):
    """Analysiert die gespeicherten Scans eines Patienten (ohne Upload)."""

    directory = patient_dir(patient_id)

    paths = {
        sequence: directory / f"{sequence}.nii.gz"
        for sequence in MSD_ORDER
    }

    missing = [s.upper() for s, path in paths.items() if not path.exists()]

    if missing:
        raise HTTPException(
            400,
            f"Missing MRI sequences for {patient_id}: {', '.join(missing)}."
        )

    with tempfile.TemporaryDirectory() as tmp:

        scan_path = combine_sequences(paths, Path(tmp) / "scan.nii.gz")

        return run_analysis(scan_path, patient_id, Path(tmp))


# ============================================================
# FRONTEND
# muss nach den API-Routen stehen, sonst fängt es /api/... ab
# ============================================================

app.mount(
    "/",
    StaticFiles(directory=ROOT / "frontend", html=True),
    name="frontend"
)

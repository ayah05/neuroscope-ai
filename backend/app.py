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
import io
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(ROOT / "ml"))

from tumor_features import compute_features  # noqa: E402
from report import generate_report  # noqa: E402


MODAL_APP = "neuroscope-monai"
MODAL_FUNCTION = "segment"

# MSD-Kanalreihenfolge im hochgeladenen 4D-Scan
FLAIR_CHANNEL = 0


app = FastAPI(title="NeuroScope API")


# ============================================================
# HELPERS
# ============================================================

def patient_id_from_filename(filename):
    """BRATS_457.nii.gz -> BRATS_457 (nur sichere Zeichen)."""

    stem = re.sub(r"\.nii(\.gz)?$", "", filename, flags=re.IGNORECASE)

    return re.sub(r"[^A-Za-z0-9_-]", "", stem) or "MRI scan"


def load_scan(path):
    """Prüft vor dem GPU-Aufruf, ob der Scan zum Modell passt."""

    try:
        scan = nib.load(path)
        shape = scan.shape
    except Exception:
        raise HTTPException(400, "The file is not a readable NIfTI scan.")

    if len(shape) != 4 or shape[3] != 4:
        raise HTTPException(
            400,
            f"Expected a 4D scan with 4 sequences (FLAIR, T1, T1ce, T2), "
            f"got shape {shape}."
        )

    return scan


def render_overlay(flair, segmentation):
    """
    Axiale Schicht mit dem meisten Tumor: FLAIR + farbige Segmentierung.
    Output: (PNG als data-URL, Schichtnummer)
    """

    tumor_per_slice = np.sum(segmentation > 0, axis=(0, 1))

    slice_idx = int(np.argmax(tumor_per_slice))

    background = flair[:, :, slice_idx].T

    mask = segmentation[:, :, slice_idx].T

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

    ax.imshow(background, cmap="gray", origin="lower")

    ax.imshow(
        np.ma.masked_where(mask == 0, mask),
        cmap="jet",
        alpha=0.55,
        origin="lower",
        vmin=1,
        vmax=4
    )

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
def analyze(image: UploadFile = File(...)):
    """
    Input:  multipart-Feld "image" mit 4D-NIfTI (.nii / .nii.gz)
    Output: {
        "patient_id": "BRATS_457",
        "segmentation": {"whole_tumor_ml": 73.8, "tumor_core_ml": 3.3,
                         "enhancing_tumor_ml": 3.2, "edema_ml": 70.5},
        "features": {...},        # kompletter Output von compute_features()
        "report_html": "<h1>...", # Befundbericht
        "overlay_image": "data:image/png;base64,...",
        "overlay_slice": 77,
        "research": None          # folgt mit der Amass-Integration
    }
    """

    filename = image.filename or ""

    if not re.search(r"\.nii(\.gz)?$", filename, flags=re.IGNORECASE):
        raise HTTPException(400, "Please upload a NIfTI scan (.nii or .nii.gz).")

    patient_id = patient_id_from_filename(filename)

    scan_bytes = image.file.read()

    with tempfile.TemporaryDirectory() as tmp:

        scan_path = Path(tmp) / "scan.nii.gz"
        scan_path.write_bytes(scan_bytes)

        scan = load_scan(scan_path)

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

        prediction_path = Path(tmp) / "prediction.nii.gz"
        prediction_path.write_bytes(prediction_bytes)

        # ------------------------------------------------------
        # 2. KENNZAHLEN + BERICHT
        # ------------------------------------------------------

        features = compute_features(prediction_path, patient_id)

        report_markdown = generate_report(features)

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
            report_markdown,
            extensions=["tables"]
        ),
        "overlay_image": overlay_image,
        "overlay_slice": overlay_slice,
        "research": None
    }


# ============================================================
# FRONTEND
# muss nach den API-Routen stehen, sonst fängt es /api/... ab
# ============================================================

app.mount(
    "/",
    StaticFiles(directory=ROOT / "frontend", html=True),
    name="frontend"
)

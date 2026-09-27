from pathlib import Path
import json

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT
    / "ml"
    / "data"
    / "real_patients"
)

PREDICTIONS_DIR = (
    ROOT
    / "outputs"
    / "predictions"
)

EVALUATIONS_DIR = (
    ROOT
    / "outputs"
    / "evaluations"
)

VISUALIZATIONS_DIR = (
    ROOT
    / "outputs"
    / "visualizations"
)


# ============================================================
# 2. MSD -> BRATS LABEL MAPPING
# ============================================================
#
# MSD ground truth:
#
# 0 = Background
# 1 = Edema
# 2 = Necrotic / non-enhancing tumor
# 3 = Enhancing tumor
#
# Prediction:
#
# 0 = Background
# 1 = Necrotic / non-enhancing tumor
# 2 = Edema
# 4 = Enhancing tumor
#
# Therefore:
#
# MSD 0 -> 0
# MSD 1 -> 2
# MSD 2 -> 1
# MSD 3 -> 4
#

MSD_TO_BRATS = np.array(
    [0, 2, 1, 4],
    dtype=np.uint8
)


# ============================================================
# 3. SEGMENTATION COLORS
# ============================================================
#
# Fixed colors are used instead of "jet" so that the same
# tumor region always has the same color.
#
# Label 1 -> Blue  -> Necrotic / non-enhancing tumor core
# Label 2 -> Green -> Peritumoral edema
# Label 4 -> Red   -> Enhancing tumor
#

SEGMENTATION_CMAP = ListedColormap([
    "blue",     # label 1
    "green",    # label 2
    "green",    # unused label 3
    "red",      # label 4
])

SEGMENTATION_NORM = BoundaryNorm(
    [0.5, 1.5, 2.5, 3.5, 4.5],
    SEGMENTATION_CMAP.N
)


# ============================================================
# 4. LEGEND
# ============================================================

LEGEND_ELEMENTS = [
    Patch(
        facecolor="blue",
        label="Necrotic / Non-enhancing Tumor Core"
    ),
    Patch(
        facecolor="green",
        label="Peritumoral Edema"
    ),
    Patch(
        facecolor="red",
        label="Enhancing Tumor"
    ),
]


# ============================================================
# 5. LOAD NIFTI
# ============================================================

def load_nifti(path):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    nii = nib.load(path)

    volume = nii.get_fdata()

    if not np.isfinite(volume).all():
        raise ValueError(
            f"Non-finite values found in {path.name}"
        )

    return volume


# ============================================================
# 6. LOAD EVALUATION
# ============================================================

def load_evaluation(patient_id):

    evaluation_file = (
        EVALUATIONS_DIR
        / f"{patient_id}_evaluation.json"
    )

    if not evaluation_file.exists():

        print(
            f"[INFO] No evaluation JSON found for {patient_id}"
        )

        return None

    with evaluation_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# 7. GET METRIC
# ============================================================

def get_metric(
    evaluation,
    region,
    metric
):

    if evaluation is None:
        return None

    try:

        return (
            evaluation
            ["regions"]
            [region]
            ["metrics"]
            [metric]
        )

    except (
        KeyError,
        TypeError
    ):

        return None


# ============================================================
# 8. FORMAT METRIC
# ============================================================

def format_metric(value):

    if value is None:
        return "N/A"

    return f"{value:.3f}"


# ============================================================
# 9. SHOW MRI
# ============================================================

def show_mri(
    ax,
    volume,
    slice_idx,
    title
):

    ax.imshow(
        volume[
            :,
            :,
            slice_idx
        ].T,
        cmap="gray",
        origin="lower"
    )

    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )


# ============================================================
# 10. SHOW SEGMENTATION
# ============================================================

def show_segmentation(
    ax,
    segmentation,
    slice_idx,
    title
):

    mask = (
        segmentation[
            :,
            :,
            slice_idx
        ].T
    )

    # Hide background label 0
    masked = np.ma.masked_where(
        mask == 0,
        mask
    )

    # Black background makes the segmentation easier to see
    ax.set_facecolor(
        "black"
    )

    ax.imshow(
        masked,
        cmap=SEGMENTATION_CMAP,
        norm=SEGMENTATION_NORM,
        origin="lower"
    )

    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )


# ============================================================
# 11. SHOW OVERLAY
# ============================================================

def show_overlay(
    ax,
    background,
    segmentation,
    slice_idx,
    title
):

    # MRI background
    ax.imshow(
        background[
            :,
            :,
            slice_idx
        ].T,
        cmap="gray",
        origin="lower"
    )

    # Segmentation
    mask = (
        segmentation[
            :,
            :,
            slice_idx
        ].T
    )

    # Hide background label 0
    masked = np.ma.masked_where(
        mask == 0,
        mask
    )

    ax.imshow(
        masked,
        cmap=SEGMENTATION_CMAP,
        norm=SEGMENTATION_NORM,
        alpha=0.55,
        origin="lower"
    )

    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )


# ============================================================
# 12. SHOW DIFFERENCE
# ============================================================

def show_difference(
    ax,
    background,
    prediction,
    ground_truth,
    slice_idx
):

    # MRI background
    ax.imshow(
        background[
            :,
            :,
            slice_idx
        ].T,
        cmap="gray",
        origin="lower"
    )

    # Every voxel where prediction and ground truth differ
    difference = (
        prediction != ground_truth
    )

    difference_slice = (
        difference[
            :,
            :,
            slice_idx
        ].T
    )

    masked = np.ma.masked_where(
        difference_slice == 0,
        difference_slice
    )

    ax.imshow(
        masked,
        cmap="Reds",
        alpha=0.65,
        origin="lower"
    )

    ax.set_title(
        "Prediction vs Ground Truth"
    )

    ax.axis(
        "off"
    )


# ============================================================
# 13. VISUALIZE ONE PATIENT
# ============================================================

def visualize_patient(
    patient_id,
    patient_dir,
    prediction_path
):

    print("\n")
    print("=" * 70)
    print(
        f"VISUALIZING PATIENT: {patient_id}"
    )
    print("=" * 70)


    # ========================================================
    # LOAD MRI MODALITIES
    # ========================================================

    t1ce = load_nifti(
        patient_dir
        / "t1ce.nii.gz"
    )

    t1 = load_nifti(
        patient_dir
        / "t1.nii.gz"
    )

    t2 = load_nifti(
        patient_dir
        / "t2.nii.gz"
    )

    flair = load_nifti(
        patient_dir
        / "flair.nii.gz"
    )


    # ========================================================
    # LOAD PREDICTION
    # ========================================================

    prediction = load_nifti(
        prediction_path
    ).astype(
        np.uint8
    )


    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    ground_truth_raw = load_nifti(
        patient_dir
        / "ground_truth.nii.gz"
    ).astype(
        np.uint8
    )


    # ========================================================
    # VALIDATE PREDICTION
    # ========================================================

    if not np.isin(
        prediction,
        [0, 1, 2, 4]
    ).all():

        raise ValueError(
            "Unexpected prediction labels: "
            f"{np.unique(prediction).tolist()}"
        )


    # ========================================================
    # VALIDATE MSD GROUND TRUTH
    # ========================================================

    if not np.isin(
        ground_truth_raw,
        [0, 1, 2, 3]
    ).all():

        raise ValueError(
            "Unexpected MSD ground-truth labels: "
            f"{np.unique(ground_truth_raw).tolist()}"
        )


    # ========================================================
    # CONVERT MSD GROUND TRUTH TO PREDICTION LABEL SCHEME
    # ========================================================

    ground_truth = (
        MSD_TO_BRATS[
            ground_truth_raw
        ]
    )


    # ========================================================
    # SHAPE CHECK
    # ========================================================

    shapes = {
        t1ce.shape,
        t1.shape,
        t2.shape,
        flair.shape,
        prediction.shape,
        ground_truth.shape,
    }

    if len(shapes) != 1:

        raise ValueError(
            f"Shape mismatch for {patient_id}: "
            f"{shapes}"
        )


    print(
        "Shape:",
        prediction.shape
    )

    print(
        "Prediction labels:",
        np.unique(prediction)
    )

    print(
        "Raw ground truth labels:",
        np.unique(ground_truth_raw)
    )

    print(
        "Mapped ground truth labels:",
        np.unique(ground_truth)
    )


    # ========================================================
    # FIND SLICE WITH MOST GROUND-TRUTH TUMOR
    # ========================================================

    tumor_per_slice = np.sum(
        ground_truth > 0,
        axis=(0, 1)
    )

    slice_idx = int(
        np.argmax(
            tumor_per_slice
        )
    )

    print(
        "Selected slice:",
        slice_idx
    )

    print(
        "Tumor voxels on slice:",
        int(
            tumor_per_slice[
                slice_idx
            ]
        )
    )


    # ========================================================
    # LOAD EVALUATION RESULTS
    # ========================================================

    evaluation = load_evaluation(
        patient_id
    )


    wt_dice = get_metric(
        evaluation,
        "whole_tumor",
        "dice"
    )

    tc_dice = get_metric(
        evaluation,
        "tumor_core",
        "dice"
    )

    et_dice = get_metric(
        evaluation,
        "enhancing_tumor",
        "dice"
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 14)
    )


    # ========================================================
    # ROW 1 - MRI MODALITIES
    # ========================================================

    show_mri(
        axes[0, 0],
        t1ce,
        slice_idx,
        "T1ce"
    )

    show_mri(
        axes[0, 1],
        t1,
        slice_idx,
        "T1"
    )

    show_mri(
        axes[0, 2],
        t2,
        slice_idx,
        "T2"
    )


    # ========================================================
    # ROW 2 - FLAIR / PREDICTION / GROUND TRUTH
    # ========================================================

    show_mri(
        axes[1, 0],
        flair,
        slice_idx,
        "FLAIR"
    )

    show_segmentation(
        axes[1, 1],
        prediction,
        slice_idx,
        "MONAI Prediction"
    )

    show_segmentation(
        axes[1, 2],
        ground_truth,
        slice_idx,
        "Ground Truth"
    )


    # ========================================================
    # ROW 3 - OVERLAYS / DIFFERENCE
    # ========================================================

    show_overlay(
        axes[2, 0],
        flair,
        prediction,
        slice_idx,
        "FLAIR + MONAI Prediction"
    )

    show_overlay(
        axes[2, 1],
        flair,
        ground_truth,
        slice_idx,
        "FLAIR + Ground Truth"
    )

    show_difference(
        axes[2, 2],
        flair,
        prediction,
        ground_truth,
        slice_idx
    )


    # ========================================================
    # TITLE + DICE SCORES
    # ========================================================

    title = (
        f"NeuroScope AI – {patient_id} – "
        f"Slice {slice_idx}\n"
        f"WT Dice: {format_metric(wt_dice)}   |   "
        f"TC Dice: {format_metric(tc_dice)}   |   "
        f"ET Dice: {format_metric(et_dice)}"
    )

    fig.suptitle(
        title,
        fontsize=17
    )


    # ========================================================
    # SEGMENTATION LEGEND
    # ========================================================

    fig.legend(
        handles=LEGEND_ELEMENTS,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.015)
    )


    # Leave space at the top for title and bottom for legend
    plt.tight_layout(
        rect=[
            0,
            0.07,
            1,
            0.93
        ]
    )


    # ========================================================
    # SAVE VISUALIZATION
    # ========================================================

    VISUALIZATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        VISUALIZATIONS_DIR
        / f"{patient_id}_visualization.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        "\nVisualization saved:"
    )

    print(
        output_file
    )


    # ========================================================
    # SHOW
    # ========================================================

    plt.show()

    # Important when processing multiple patients
    plt.close(fig)


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("NEUROSCOPE VISUALIZATION")
    print("=" * 70)


    # ========================================================
    # CHECK DIRECTORIES
    # ========================================================

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Patient directory not found:\n"
            f"{DATA_DIR}"
        )


    if not PREDICTIONS_DIR.exists():

        raise FileNotFoundError(
            f"Prediction directory not found:\n"
            f"{PREDICTIONS_DIR}\n\n"
            "Run modal_inference.py first."
        )


    VISUALIZATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # FIND PATIENTS
    # ========================================================

    patient_dirs = sorted(
        directory
        for directory
        in DATA_DIR.iterdir()
        if directory.is_dir()
    )

    print(
        f"\nPatients found: "
        f"{len(patient_dirs)}"
    )

    if not patient_dirs:

        raise RuntimeError(
            "No patient directories found."
        )


    # ========================================================
    # STATUS
    # ========================================================

    successful = []
    skipped = []
    failed = []


    # ========================================================
    # PROCESS ALL PATIENTS
    # ========================================================

    for patient_dir in patient_dirs:

        patient_id = (
            patient_dir.name
        )

        prediction_path = (
            PREDICTIONS_DIR
            / f"{patient_id}_prediction.nii.gz"
        )

        ground_truth_path = (
            patient_dir
            / "ground_truth.nii.gz"
        )


        # ----------------------------------------------------
        # Prediction missing
        # ----------------------------------------------------

        if not prediction_path.exists():

            print(
                f"\n[SKIP] {patient_id}: "
                "prediction missing."
            )

            skipped.append(
                patient_id
            )

            continue


        # ----------------------------------------------------
        # Ground truth missing
        # ----------------------------------------------------

        if not ground_truth_path.exists():

            print(
                f"\n[SKIP] {patient_id}: "
                "ground truth missing."
            )

            skipped.append(
                patient_id
            )

            continue


        # ----------------------------------------------------
        # Visualize
        # ----------------------------------------------------

        try:

            visualize_patient(
                patient_id,
                patient_dir,
                prediction_path
            )

            successful.append(
                patient_id
            )


        except Exception as error:

            print(
                f"\n[ERROR] "
                f"{patient_id}: "
                f"{error}"
            )

            failed.append({
                "patient_id":
                    patient_id,

                "error":
                    str(error)
            })


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Successful: "
        f"{len(successful)}"
    )

    print(
        f"Skipped:    "
        f"{len(skipped)}"
    )

    print(
        f"Failed:     "
        f"{len(failed)}"
    )


    if successful:

        print(
            "\nSuccessfully visualized:"
        )

        for patient_id in successful:

            print(
                f"  ✓ {patient_id}"
            )


    if skipped:

        print(
            "\nSkipped:"
        )

        for patient_id in skipped:

            print(
                f"  - {patient_id}"
            )


    if failed:

        print(
            "\nFailed patients:"
        )

        for item in failed:

            print(
                f"  ✗ "
                f"{item['patient_id']}: "
                f"{item['error']}"
            )


    print(
        "\nVisualizations:"
    )

    print(
        VISUALIZATIONS_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
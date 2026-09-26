from pathlib import Path

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent.parent

DATA = ROOT / "ml" / "data" / "real_patient"
PREDICTION = ROOT / "outputs" / "brats_457_prediction.nii.gz"


# ============================================================
# 1. LOAD DATA
# ============================================================

t1ce = nib.load(DATA / "patient_t1ce.nii.gz").get_fdata()
t1 = nib.load(DATA / "patient_t1.nii.gz").get_fdata()
t2 = nib.load(DATA / "patient_t2.nii.gz").get_fdata()
flair = nib.load(DATA / "patient_flair.nii.gz").get_fdata()

prediction = nib.load(PREDICTION).get_fdata()
ground_truth = nib.load(DATA / "ground_truth.nii.gz").get_fdata()

# MSD-Labels (1=Ödem, 2=nicht-anreichernd, 3=anreichernd) auf das
# BraTS-Schema der Prediction umrechnen (2=Ödem, 1=nicht-anreichernd, 4=anreichernd)
MSD_TO_BRATS = np.array([0, 2, 1, 4])
ground_truth = MSD_TO_BRATS[ground_truth.astype(np.uint8)]


print("\n========== SHAPES ==========")

print("T1ce:        ", t1ce.shape)
print("T1:          ", t1.shape)
print("T2:          ", t2.shape)
print("FLAIR:       ", flair.shape)
print("Prediction:  ", prediction.shape)
print("Ground Truth:", ground_truth.shape)


print("\n========== LABELS ==========")

print("Prediction labels:", np.unique(prediction))
print("Ground truth labels:", np.unique(ground_truth))


# ============================================================
# 2. FIND SLICE WITH MOST GROUND-TRUTH TUMOR
# ============================================================

tumor_per_slice = np.sum(
    ground_truth > 0,
    axis=(0, 1)
)

slice_idx = int(np.argmax(tumor_per_slice))

print("\nSlice with most tumor:", slice_idx)
print(
    "Tumor voxels on slice:",
    tumor_per_slice[slice_idx]
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def show_mri(ax, volume, title):

    ax.imshow(
        volume[:, :, slice_idx].T,
        cmap="gray",
        origin="lower"
    )

    ax.set_title(title)
    ax.axis("off")


def show_overlay(ax, background, segmentation, title):

    # MRI
    ax.imshow(
        background[:, :, slice_idx].T,
        cmap="gray",
        origin="lower"
    )

    # Segmentation
    mask = segmentation[:, :, slice_idx].T

    masked = np.ma.masked_where(
        mask == 0,
        mask
    )

    ax.imshow(
        masked,
        cmap="jet",
        alpha=0.55,
        origin="lower",
        vmin=1,
        vmax=4
    )

    ax.set_title(title)
    ax.axis("off")


# ============================================================
# 4. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(15, 14)
)


# ------------------------------------------------------------
# MRI modalities
# ------------------------------------------------------------

show_mri(
    axes[0, 0],
    t1ce,
    "T1ce"
)

show_mri(
    axes[0, 1],
    t1,
    "T1"
)

show_mri(
    axes[0, 2],
    t2,
    "T2"
)


show_mri(
    axes[1, 0],
    flair,
    "FLAIR"
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

axes[1, 1].imshow(
    prediction[:, :, slice_idx].T,
    cmap="jet",
    origin="lower",
    vmin=0,
    vmax=4
)

axes[1, 1].set_title(
    "MONAI Prediction"
)

axes[1, 1].axis("off")


# ------------------------------------------------------------
# Ground Truth
# ------------------------------------------------------------

axes[1, 2].imshow(
    ground_truth[:, :, slice_idx].T,
    cmap="jet",
    origin="lower",
    vmin=0,
    vmax=4
)

axes[1, 2].set_title(
    "Ground Truth"
)

axes[1, 2].axis("off")


# ------------------------------------------------------------
# Prediction overlay
# ------------------------------------------------------------

show_overlay(
    axes[2, 0],
    flair,
    prediction,
    "FLAIR + MONAI Prediction"
)


# ------------------------------------------------------------
# Ground truth overlay
# ------------------------------------------------------------

show_overlay(
    axes[2, 1],
    flair,
    ground_truth,
    "FLAIR + Ground Truth"
)


# ------------------------------------------------------------
# Difference
# ------------------------------------------------------------

difference = (
    prediction != ground_truth
).astype(np.uint8)

axes[2, 2].imshow(
    flair[:, :, slice_idx].T,
    cmap="gray",
    origin="lower"
)

difference_mask = np.ma.masked_where(
    difference[:, :, slice_idx].T == 0,
    difference[:, :, slice_idx].T
)

axes[2, 2].imshow(
    difference_mask,
    cmap="Reds",
    alpha=0.6,
    origin="lower"
)

axes[2, 2].set_title(
    "Prediction vs Ground Truth"
)

axes[2, 2].axis("off")


# ============================================================
# 5. SAVE
# ============================================================

plt.suptitle(
    f"NeuroScope AI – BRATS_457 – Slice {slice_idx}",
    fontsize=18
)

plt.tight_layout()


output_file = (
    ROOT
    / "outputs"
    / "brats_457_visualization.png"
)

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

print("\nVisualization saved:")
print(output_file)


plt.show()
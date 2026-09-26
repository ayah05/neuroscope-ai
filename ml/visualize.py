from pathlib import Path

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent.parent

DATA = ROOT / "ml" / "data" / "test_patient"
SEGMENTATION = ROOT / "outputs" / "segmentation.nii.gz"


# --------------------------------------------------
# 1. MRI-DATEN LADEN
# --------------------------------------------------

t1ce = nib.load(DATA / "test_t1ce.nii.gz").get_fdata()
t1 = nib.load(DATA / "test_t1.nii.gz").get_fdata()
t2 = nib.load(DATA / "test_t2.nii.gz").get_fdata()
flair = nib.load(DATA / "test_flair.nii.gz").get_fdata()

seg = nib.load(SEGMENTATION).get_fdata()


print("T1ce:", t1ce.shape)
print("T1:", t1.shape)
print("T2:", t2.shape)
print("FLAIR:", flair.shape)
print("Segmentation:", seg.shape)

print("Segmentation labels:", np.unique(seg))


# --------------------------------------------------
# 2. SLICE AUSWÄHLEN
# --------------------------------------------------

# Falls Tumor vorhanden:
tumor_per_slice = np.sum(seg > 0, axis=(0, 1))

if tumor_per_slice.max() > 0:
    slice_idx = int(np.argmax(tumor_per_slice))
else:
    slice_idx = flair.shape[2] // 2

print("Showing slice:", slice_idx)


# --------------------------------------------------
# 3. VISUALISIERUNG
# --------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 9))


def show_mri(ax, volume, title):
    ax.imshow(
        volume[:, :, slice_idx].T,
        cmap="gray",
        origin="lower"
    )

    ax.set_title(title)
    ax.axis("off")


show_mri(axes[0, 0], t1ce, "T1ce")
show_mri(axes[0, 1], t1, "T1")
show_mri(axes[0, 2], t2, "T2")

show_mri(axes[1, 0], flair, "FLAIR")


# --------------------------------------------------
# SEGMENTATION
# --------------------------------------------------

axes[1, 1].imshow(
    seg[:, :, slice_idx].T,
    cmap="viridis",
    origin="lower",
    vmin=0,
    vmax=4
)

axes[1, 1].set_title("MONAI Segmentation")
axes[1, 1].axis("off")


# --------------------------------------------------
# FLAIR + SEGMENTATION OVERLAY
# --------------------------------------------------

axes[1, 2].imshow(
    flair[:, :, slice_idx].T,
    cmap="gray",
    origin="lower"
)

mask = np.ma.masked_where(
    seg[:, :, slice_idx].T == 0,
    seg[:, :, slice_idx].T
)

axes[1, 2].imshow(
    mask,
    cmap="jet",
    alpha=0.6,
    origin="lower",
    vmin=1,
    vmax=4
)

axes[1, 2].set_title("FLAIR + Tumor Overlay")
axes[1, 2].axis("off")


plt.suptitle(
    f"NeuroScope AI – Slice {slice_idx}",
    fontsize=16
)

plt.tight_layout()

# zusätzlich als PNG speichern
OUTPUT = ROOT / "outputs" / "visualization.png"

plt.savefig(
    OUTPUT,
    dpi=150,
    bbox_inches="tight"
)

print("Visualization saved:", OUTPUT)

plt.show()
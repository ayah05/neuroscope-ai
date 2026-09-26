from pathlib import Path
import json

import nibabel as nib
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).parent.parent

DATA = ROOT / "ml" / "data" / "real_patient"

PREDICTION = (
    ROOT
    / "outputs"
    / "brats_457_prediction.nii.gz"
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading MRI data...")

prediction_nii = nib.load(PREDICTION)

prediction = prediction_nii.get_fdata()

t1ce = nib.load(
    DATA / "patient_t1ce.nii.gz"
).get_fdata()

t1 = nib.load(
    DATA / "patient_t1.nii.gz"
).get_fdata()

t2 = nib.load(
    DATA / "patient_t2.nii.gz"
).get_fdata()

flair = nib.load(
    DATA / "patient_flair.nii.gz"
).get_fdata()


# ============================================================
# 2. VOXEL INFORMATION
# ============================================================

voxel_spacing = prediction_nii.header.get_zooms()[:3]

voxel_volume_mm3 = float(
    np.prod(voxel_spacing)
)


print("\n========== VOXEL INFO ==========")

print(
    "Voxel spacing:",
    voxel_spacing
)

print(
    "Voxel volume:",
    voxel_volume_mm3,
    "mm³"
)


# ============================================================
# 3. CREATE TUMOR REGIONS
# ============================================================

# BraTS label map used in our final segmentation:
#
# 0 = Background
# 1 = Necrotic / non-enhancing tumor core
# 2 = Peritumoral edema
# 4 = Enhancing tumor


necrotic_non_enhancing = (
    prediction == 1
)

edema = (
    prediction == 2
)

enhancing_tumor = (
    prediction == 4
)


# Tumor Core:
# necrotic/non-enhancing + enhancing tumor

tumor_core = np.logical_or(
    necrotic_non_enhancing,
    enhancing_tumor
)


# Whole Tumor:
# every tumor-associated label

whole_tumor = (
    prediction > 0
)


# ============================================================
# 4. VOLUME CALCULATION
# ============================================================

def volume_ml(mask):

    voxel_count = np.count_nonzero(
        mask
    )

    volume_mm3 = (
        voxel_count
        * voxel_volume_mm3
    )

    return float(
        volume_mm3 / 1000
    )


whole_tumor_volume = volume_ml(
    whole_tumor
)

tumor_core_volume = volume_ml(
    tumor_core
)

enhancing_tumor_volume = volume_ml(
    enhancing_tumor
)

edema_volume = volume_ml(
    edema
)


# ============================================================
# 5. FRACTIONS
# ============================================================

if whole_tumor_volume > 0:

    enhancing_fraction = (
        enhancing_tumor_volume
        / whole_tumor_volume
    )

    core_fraction = (
        tumor_core_volume
        / whole_tumor_volume
    )

    edema_fraction = (
        edema_volume
        / whole_tumor_volume
    )

else:

    enhancing_fraction = 0.0
    core_fraction = 0.0
    edema_fraction = 0.0


# ============================================================
# 6. CONNECTED COMPONENTS
# ============================================================

# This estimates whether the segmentation consists of
# one or multiple spatially separated tumor regions.

try:

    from scipy.ndimage import label

    labeled_tumor, lesion_count = label(
        whole_tumor
    )

    lesion_count = int(
        lesion_count
    )

except ImportError:

    lesion_count = None


# ============================================================
# 7. TUMOR CENTER
# ============================================================

tumor_coordinates = np.argwhere(
    whole_tumor
)

if len(tumor_coordinates) > 0:

    centroid_voxel = np.mean(
        tumor_coordinates,
        axis=0
    )

    centroid_voxel = [
        round(float(x), 1)
        for x in centroid_voxel
    ]

else:

    centroid_voxel = None


# ============================================================
# 8. MRI INTENSITY FEATURES
# ============================================================

def calculate_intensity_features(
    volume,
    mask
):

    values = volume[mask]

    if values.size == 0:

        return {
            "mean": None,
            "std": None,
            "median": None
        }

    return {

        "mean": round(
            float(np.mean(values)),
            2
        ),

        "std": round(
            float(np.std(values)),
            2
        ),

        "median": round(
            float(np.median(values)),
            2
        )
    }


intensity_features = {

    "T1": calculate_intensity_features(
        t1,
        whole_tumor
    ),

    "T1ce": calculate_intensity_features(
        t1ce,
        whole_tumor
    ),

    "T2": calculate_intensity_features(
        t2,
        whole_tumor
    ),

    "FLAIR": calculate_intensity_features(
        flair,
        whole_tumor
    )
}


# ============================================================
# 9. CREATE STRUCTURED FEATURE OBJECT
# ============================================================

features = {

    "patient_id":
        "BRATS_457",

    "analysis_type":
        "automated MRI segmentation",

    "model":
        "MONAI SegResNet",

    "tumor_context":
        "brain tumor / glioma MRI segmentation",

    "segmentation": {

        "whole_tumor_volume_ml":
            round(
                whole_tumor_volume,
                2
            ),

        "tumor_core_volume_ml":
            round(
                tumor_core_volume,
                2
            ),

        "enhancing_tumor_volume_ml":
            round(
                enhancing_tumor_volume,
                2
            ),

        "edema_volume_ml":
            round(
                edema_volume,
                2
            )
    },

    "ratios": {

        "enhancing_fraction":
            round(
                enhancing_fraction,
                3
            ),

        "core_fraction":
            round(
                core_fraction,
                3
            ),

        "edema_fraction":
            round(
                edema_fraction,
                3
            )
    },

    "spatial": {

        "lesion_count":
            lesion_count,

        "centroid_voxel":
            centroid_voxel
    },

    "mri_intensity": intensity_features,

    "classification":
        None
}


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print(
    "\n========== TUMOR FEATURES =========="
)

print(
    json.dumps(
        features,
        indent=2
    )
)


# ============================================================
# 11. SAVE JSON
# ============================================================

output_file = (
    ROOT
    / "outputs"
    / "brats_457_features.json"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        features,
        file,
        indent=2
    )


print(
    "\nFeatures saved to:"
)

print(
    output_file
)
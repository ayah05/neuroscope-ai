from pathlib import Path
import json

import nibabel as nib
import numpy as np
from scipy.ndimage import label


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).parent.parent

PREDICTION = (
    ROOT
    / "outputs"
    / "brats_457_prediction.nii.gz"
)

OUTPUT_FILE = (
    ROOT
    / "outputs"
    / "brats_457_features.json"
)


# Zusammenhängende Regionen unter dieser Größe gelten als
# Segmentierungsrauschen und zählen nicht als eigene Läsion
# (1000 Voxel = 1 ml bei 1 mm Auflösung).
MIN_LESION_VOXELS = 1000


# ============================================================
# HELPERS
# ============================================================

def volume_ml(mask, voxel_volume_mm3):

    voxel_count = np.count_nonzero(mask)

    return float(voxel_count * voxel_volume_mm3 / 1000)


def fraction(part, whole):

    if whole == 0:
        return 0.0

    return round(part / whole, 3)


def count_lesions(mask):

    labeled, component_count = label(mask)

    if component_count == 0:
        return 0

    # bincount[0] ist der Hintergrund
    sizes = np.bincount(labeled.ravel())[1:]

    return int(np.sum(sizes >= MIN_LESION_VOXELS))


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def compute_features(
    prediction_path: Path,
    patient_id: str
) -> dict:
    """
    Input:  Pfad zu einer Segmentierung im BraTS-Schema (.nii.gz)
    Output: {
        "patient_id": "BRATS_457",
        ...
        "segmentation": {"whole_tumor_volume_ml": 43.7, ...},
        "ratios": {"enhancing_fraction": 0.169, ...},
        "spatial": {"lesion_count": 1, "centroid_voxel": [...]},
        "classification": None
    }
    """

    prediction_nii = nib.load(prediction_path)

    prediction = prediction_nii.get_fdata()

    # --------------------------------------------------------
    # 1. VOXEL INFORMATION
    # --------------------------------------------------------

    voxel_spacing = prediction_nii.header.get_zooms()[:3]

    voxel_volume_mm3 = float(np.prod(voxel_spacing))

    # --------------------------------------------------------
    # 2. TUMOR REGIONS
    #
    # BraTS label map used in our final segmentation:
    #
    # 0 = Background
    # 1 = Necrotic / non-enhancing tumor core
    # 2 = Peritumoral edema
    # 4 = Enhancing tumor
    # --------------------------------------------------------

    necrotic_non_enhancing = prediction == 1
    edema = prediction == 2
    enhancing_tumor = prediction == 4

    # Tumor Core: necrotic/non-enhancing + enhancing tumor
    tumor_core = np.logical_or(
        necrotic_non_enhancing,
        enhancing_tumor
    )

    # Whole Tumor: every tumor-associated label
    whole_tumor = prediction > 0

    # --------------------------------------------------------
    # 3. VOLUMES
    # --------------------------------------------------------

    whole_tumor_volume = volume_ml(whole_tumor, voxel_volume_mm3)
    tumor_core_volume = volume_ml(tumor_core, voxel_volume_mm3)
    enhancing_tumor_volume = volume_ml(enhancing_tumor, voxel_volume_mm3)
    edema_volume = volume_ml(edema, voxel_volume_mm3)

    # --------------------------------------------------------
    # 4. TUMOR CENTER
    # --------------------------------------------------------

    tumor_coordinates = np.argwhere(whole_tumor)

    if len(tumor_coordinates) > 0:

        centroid_voxel = [
            round(float(x), 1)
            for x in tumor_coordinates.mean(axis=0)
        ]

    else:

        centroid_voxel = None

    # --------------------------------------------------------
    # 5. STRUCTURED FEATURE OBJECT
    # --------------------------------------------------------

    return {

        "patient_id": patient_id,

        "analysis_type": "automated MRI segmentation",

        "model": "MONAI SegResNet",

        "tumor_context": "brain tumor / glioma MRI segmentation",

        "segmentation": {
            "whole_tumor_volume_ml": round(whole_tumor_volume, 2),
            "tumor_core_volume_ml": round(tumor_core_volume, 2),
            "enhancing_tumor_volume_ml": round(enhancing_tumor_volume, 2),
            "edema_volume_ml": round(edema_volume, 2)
        },

        "ratios": {
            "enhancing_fraction": fraction(
                enhancing_tumor_volume, whole_tumor_volume
            ),
            "core_fraction": fraction(
                tumor_core_volume, whole_tumor_volume
            ),
            "edema_fraction": fraction(
                edema_volume, whole_tumor_volume
            )
        },

        "spatial": {
            "lesion_count": count_lesions(whole_tumor),
            "centroid_voxel": centroid_voxel
        },

        "classification": None
    }


# ============================================================
# LOKALER TEST
# ============================================================

if __name__ == "__main__":

    print("\nLoading segmentation...")

    features = compute_features(
        PREDICTION,
        patient_id="BRATS_457"
    )

    print("\n========== TUMOR FEATURES ==========")

    print(json.dumps(features, indent=2))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(features, file, indent=2)

    print("\nFeatures saved to:")
    print(OUTPUT_FILE)

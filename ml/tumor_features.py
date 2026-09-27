from pathlib import Path
import json

import nibabel as nib
import numpy as np


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

FEATURES_DIR = (
    ROOT
    / "outputs"
    / "features"
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_aligned_mri(path, reference):
    """
    Load MRI and verify that it uses the same spatial grid
    as the segmentation.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"MRI file not found: {path}"
        )

    nii = nib.load(path)

    if nii.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch for {path.name}: "
            f"{nii.shape} != {reference.shape}"
        )

    if not np.allclose(
        nii.affine,
        reference.affine,
        rtol=1e-5,
        atol=1e-4,
    ):
        raise ValueError(
            f"Spatial alignment mismatch for {path.name}"
        )

    volume = nii.get_fdata(
        dtype=np.float32
    )

    if not np.isfinite(volume).all():
        raise ValueError(
            f"Non-finite MRI values in {path.name}"
        )

    return volume


def volume_ml(mask, voxel_volume_mm3):
    """
    Convert the number of masked voxels to millilitres.
    """

    return float(
        np.count_nonzero(mask)
        * voxel_volume_mm3
        / 1000.0
    )


def fraction(numerator, denominator):
    """
    Return None when the ratio is undefined.
    """

    if denominator == 0:
        return None

    return round(
        float(numerator / denominator),
        4
    )


def intensity_features(volume, mask):
    """
    Calculate raw MRI intensity statistics inside
    the supplied segmentation mask.

    MRI intensities are not standardized across patients.
    """

    values = volume[mask]

    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
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
        ),
    }


def component_features(
    mask,
    voxel_volume_mm3
):
    """
    Count connected mask components using 26-connectivity.

    These are segmentation components, not confirmed lesions.

    No small-component filtering is applied.
    """

    try:
        from scipy.ndimage import label

    except ImportError:

        return {
            "available": False,
            "reason": "scipy is not installed",
            "connectivity": 26,
            "connected_component_count": None,
            "component_volumes_ml": None,
        }

    labeled, count = label(
        mask,
        structure=np.ones(
            (3, 3, 3),
            dtype=np.uint8
        ),
    )

    # Index 0 represents background
    sizes = np.bincount(
        labeled.ravel()
    )[1:]

    sizes = np.sort(
        sizes
    )[::-1]

    volumes = (
        sizes
        * voxel_volume_mm3
        / 1000.0
    )

    return {
        "available": True,
        "connectivity": 26,
        "small_component_filter_applied": False,
        "connected_component_count": int(count),
        "component_volumes_ml": [
            round(
                float(value),
                4
            )
            for value in volumes
        ],
    }


# ============================================================
# 3. PROCESS ONE PATIENT
# ============================================================

def process_patient(
    patient_id,
    patient_dir,
    prediction_file
):

    print("\n")
    print("=" * 70)
    print(f"PROCESSING PATIENT: {patient_id}")
    print("=" * 70)

    print(
        "\nPrediction:",
        prediction_file
    )

    print(
        "MRI directory:",
        patient_dir
    )

    # --------------------------------------------------------
    # Load segmentation
    # --------------------------------------------------------

    print(
        "\nLoading segmentation..."
    )

    prediction_nii = nib.load(
        prediction_file
    )

    if len(
        prediction_nii.shape
    ) != 3:

        raise ValueError(
            f"Expected a 3D segmentation, "
            f"got {prediction_nii.shape}"
        )

    prediction = (
        prediction_nii
        .get_fdata()
    )

    if not np.isfinite(
        prediction
    ).all():

        raise ValueError(
            "Segmentation contains non-finite values"
        )

    # --------------------------------------------------------
    # Validate BraTS labels
    # --------------------------------------------------------

    allowed_labels = [
        0,
        1,
        2,
        4
    ]

    if not np.isin(
        prediction,
        allowed_labels
    ).all():

        raise ValueError(
            "Unexpected segmentation labels: "
            f"{np.unique(prediction).tolist()}"
        )

    prediction = prediction.astype(
        np.uint8
    )

    print(
        "Segmentation shape:",
        prediction.shape
    )

    print(
        "Labels:",
        np.unique(prediction)
    )


    # ========================================================
    # 4. SPATIAL INFORMATION
    # ========================================================

    affine = prediction_nii.affine

    if not np.isfinite(
        affine
    ).all():

        raise ValueError(
            "Segmentation affine contains invalid values"
        )

    spatial_unit, _ = (
        prediction_nii
        .header
        .get_xyzt_units()
    )

    # Explicit conversion to millimetres
    unit_to_mm = {
        "mm": 1.0,
        "meter": 1000.0,
        "micron": 0.001,
    }

    if spatial_unit not in unit_to_mm:

        raise ValueError(
            f"Spatial unit is '{spatial_unit}'. "
            "Verify the NIfTI spatial unit "
            "before calculating volumes."
        )

    scale_to_mm = (
        unit_to_mm[
            spatial_unit
        ]
    )

    voxel_spacing_mm = (
        nib.affines.voxel_sizes(
            affine
        )
        * scale_to_mm
    )

    # Determinant also handles affine grids containing shear
    voxel_volume_mm3 = float(
        abs(
            np.linalg.det(
                affine[:3, :3]
            )
        )
        * scale_to_mm ** 3
    )

    if (
        not np.isfinite(
            voxel_volume_mm3
        )
        or voxel_volume_mm3 <= 0
    ):

        raise ValueError(
            "Invalid voxel volume"
        )

    print(
        "Voxel spacing (mm):",
        voxel_spacing_mm
    )

    print(
        "Voxel volume (mm³):",
        voxel_volume_mm3
    )


    # ========================================================
    # 5. LOAD MRI MODALITIES
    # ========================================================

    print(
        "\nLoading and checking MRI modalities..."
    )

    modalities = {

        "T1": load_aligned_mri(
            patient_dir
            / "t1.nii.gz",
            prediction_nii,
        ),

        "T1ce": load_aligned_mri(
            patient_dir
            / "t1ce.nii.gz",
            prediction_nii,
        ),

        "T2": load_aligned_mri(
            patient_dir
            / "t2.nii.gz",
            prediction_nii,
        ),

        "FLAIR": load_aligned_mri(
            patient_dir
            / "flair.nii.gz",
            prediction_nii,
        ),
    }

    print(
        "MRI modalities successfully loaded."
    )


    # ========================================================
    # 6. BRATS REGIONS
    #
    # Final predicted label map:
    #
    # 0 = Background
    # 1 = Necrotic / non-enhancing tumor core
    # 2 = Peritumoral edema
    # 4 = Enhancing tumor
    # ========================================================

    regions = {

        "necrotic_non_enhancing":
            prediction == 1,

        "edema":
            prediction == 2,

        "enhancing_tumor":
            prediction == 4,

        "tumor_core":
            np.isin(
                prediction,
                [1, 4]
            ),

        "whole_tumor":
            prediction > 0,
    }

    whole_tumor = (
        regions[
            "whole_tumor"
        ]
    )


    # ========================================================
    # 7. VOXEL COUNTS
    # ========================================================

    counts = {

        name:
            int(
                np.count_nonzero(
                    mask
                )
            )

        for name, mask
        in regions.items()
    }


    # ========================================================
    # 8. VOLUMES
    # ========================================================

    volumes = {

        name:
            volume_ml(
                mask,
                voxel_volume_mm3
            )

        for name, mask
        in regions.items()
    }


    print(
        "\nTumor volumes:"
    )

    for name, value in volumes.items():

        print(
            f"  {name}: "
            f"{value:.4f} ml"
        )


    # ========================================================
    # 9. CENTROID
    # ========================================================

    centroid_voxel = None
    centroid_world_mm = None

    if counts[
        "whole_tumor"
    ] > 0:

        coordinates = np.argwhere(
            whole_tumor
        )

        center = (
            coordinates
            .mean(axis=0)
        )

        world_center_mm = (
            nib.affines.apply_affine(
                affine,
                center
            )
            * scale_to_mm
        )

        centroid_voxel = [
            round(
                float(value),
                3
            )
            for value in center
        ]

        centroid_world_mm = [
            round(
                float(value),
                3
            )
            for value in world_center_mm
        ]


    # ========================================================
    # 10. STRUCTURED FEATURE OBJECT
    # ========================================================

    features = {

        "patient_id":
            patient_id,

        "analysis_type":
            "predicted_segmentation_feature_extraction",

        "model": {
            "name":
                "MONAI SegResNet",

            "bundle":
                "brats_mri_segmentation",
        },

        "source_files": {

            "prediction":
                prediction_file.name,

            "t1":
                "t1.nii.gz",

            "t1ce":
                "t1ce.nii.gz",

            "t2":
                "t2.nii.gz",

            "flair":
                "flair.nii.gz",
        },

        "label_mapping": {

            "0":
                "background",

            "1":
                "necrotic_non_enhancing_tumor_core",

            "2":
                "peritumoral_edema",

            "4":
                "enhancing_tumor",
        },

        "geometry": {

            "shape":
                list(
                    prediction.shape
                ),

            "voxel_spacing_mm": [
                round(
                    float(value),
                    6
                )
                for value
                in voxel_spacing_mm
            ],

            "voxel_volume_mm3":
                voxel_volume_mm3,

            "source_spatial_unit":
                spatial_unit,
        },

        "quality_checks": {

            "allowed_labels_verified":
                True,

            "mri_shapes_match_prediction":
                True,

            "mri_affines_match_prediction":
                True,

            "finite_values_verified":
                True,

            "empty_segmentation":
                counts[
                    "whole_tumor"
                ] == 0,

            "ground_truth_evaluation_performed":
                False,
        },

        "segmentation": {

            f"{name}_volume_ml":
                round(
                    value,
                    4
                )

            for name, value
            in volumes.items()
        },

        "voxel_counts":
            counts,

        "ratios": {

            "denominator":
                "whole_tumor",

            "enhancing_fraction":
                fraction(
                    counts[
                        "enhancing_tumor"
                    ],
                    counts[
                        "whole_tumor"
                    ],
                ),

            "necrotic_non_enhancing_fraction":
                fraction(
                    counts[
                        "necrotic_non_enhancing"
                    ],
                    counts[
                        "whole_tumor"
                    ],
                ),

            "core_fraction":
                fraction(
                    counts[
                        "tumor_core"
                    ],
                    counts[
                        "whole_tumor"
                    ],
                ),

            "edema_fraction":
                fraction(
                    counts[
                        "edema"
                    ],
                    counts[
                        "whole_tumor"
                    ],
                ),
        },

        "spatial": {

            "centroid_voxel":
                centroid_voxel,

            "centroid_world_mm":
                centroid_world_mm,

            "world_coordinate_system":
                "NIfTI RAS+",

            "anatomical_location":
                None,

            "components":
                component_features(
                    whole_tumor,
                    voxel_volume_mm3,
                ),
        },

        "mri_intensity": {

            "measurement_region":
                "predicted_whole_tumor",

            "normalization":
                "none",

            "units":
                "arbitrary",

            "features": {

                name:
                    intensity_features(
                        volume,
                        whole_tumor
                    )

                for name, volume
                in modalities.items()
            },
        },

        "classification":
            None,

        "limitations": [

            "Features describe the model prediction, not verified tissue.",

            "Whole tumor includes the predicted edema region.",

            "Tumor core includes the enhancing region; these volumes overlap.",

            "Connected components are not confirmed separate lesions.",

            "Raw MRI intensities are not standardized across patients.",

            "No anatomical atlas localization or classification performed.",
        ],
    }


    # ========================================================
    # 11. SAVE JSON
    # ========================================================

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        FEATURES_DIR
        / f"{patient_id}_features.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            features,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print(
        "\nFeatures saved to:"
    )

    print(
        output_file
    )

    print(
        f"\n[DONE] {patient_id}"
    )

    return features


# ============================================================
# BACKEND ENTRY POINT (backend/app.py)
# ============================================================

# Components below this size are treated as segmentation noise
# when counting lesions for the report (1 mL = 1000 voxels at 1 mm).
MIN_LESION_ML = 1.0


def compute_features(
    prediction_path,
    patient_id,
    mri_dir
):
    """
    Input:  prediction_path – segmentation (.nii.gz, BraTS labels)
            patient_id      – shown in report and file names
            mri_dir         – folder with t1/t1ce/t2/flair.nii.gz
    Output: process_patient() features plus
            spatial.lesion_count (components >= MIN_LESION_ML)
    """

    features = process_patient(
        patient_id=patient_id,
        patient_dir=Path(mri_dir),
        prediction_file=Path(prediction_path),
    )


    components = features["spatial"]["components"]

    volumes = components.get("component_volumes_ml") or []

    lesion_count = sum(
        1 for value in volumes if value >= MIN_LESION_ML
    )

    # tumor present but only small fragments -> still one lesion
    if lesion_count == 0 and volumes:
        lesion_count = 1

    features["spatial"]["lesion_count"] = lesion_count

    features["spatial"]["lesion_count_min_volume_ml"] = MIN_LESION_ML


    return features


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("NEUROSCOPE TUMOR FEATURE EXTRACTION")
    print("=" * 70)


    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Patient data directory "
            f"does not exist:\n{DATA_DIR}"
        )

    if not PREDICTIONS_DIR.exists():

        raise FileNotFoundError(
            f"Prediction directory "
            f"does not exist:\n{PREDICTIONS_DIR}\n\n"
            "Run modal_inference.py first."
        )


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Find patients
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Process each patient
    # --------------------------------------------------------

    successful = []
    skipped = []
    failed = []


    for patient_dir in patient_dirs:

        patient_id = (
            patient_dir.name
        )

        prediction_file = (
            PREDICTIONS_DIR
            / f"{patient_id}_prediction.nii.gz"
        )


        # ----------------------------------------------------
        # Prediction missing
        # ----------------------------------------------------

        if not prediction_file.exists():

            print(
                f"\n[SKIP] {patient_id}: "
                "prediction not found."
            )

            skipped.append(
                patient_id
            )

            continue


        # ----------------------------------------------------
        # Process patient
        # ----------------------------------------------------

        try:

            process_patient(
                patient_id=
                    patient_id,

                patient_dir=
                    patient_dir,

                prediction_file=
                    prediction_file,
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

            failed.append(
                {
                    "patient_id":
                        patient_id,

                    "error":
                        str(error),
                }
            )


    # ========================================================
    # 13. SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
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
            "\nSuccessful patients:"
        )

        for patient_id in successful:

            print(
                f"  ✓ {patient_id}"
            )


    if skipped:

        print(
            "\nSkipped patients:"
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
        "\nFeature files:"
    )

    print(
        FEATURES_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
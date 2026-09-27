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

EVALUATIONS_DIR = (
    ROOT
    / "outputs"
    / "evaluations"
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def safe_divide(numerator, denominator):
    """
    Return None when the metric is undefined.
    """

    if denominator == 0:
        return None

    return float(numerator / denominator)


def evaluate_mask(prediction_mask, ground_truth_mask):
    """
    Calculate segmentation metrics for one binary region.
    """

    true_positive = int(
        np.count_nonzero(
            prediction_mask & ground_truth_mask
        )
    )

    false_positive = int(
        np.count_nonzero(
            prediction_mask & ~ground_truth_mask
        )
    )

    false_negative = int(
        np.count_nonzero(
            ~prediction_mask & ground_truth_mask
        )
    )

    prediction_count = int(
        np.count_nonzero(prediction_mask)
    )

    ground_truth_count = int(
        np.count_nonzero(ground_truth_mask)
    )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice = safe_divide(
        2 * true_positive,
        prediction_count + ground_truth_count,
    )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = safe_divide(
        true_positive,
        true_positive
        + false_positive
        + false_negative,
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    # --------------------------------------------------------
    # Relative volume difference
    #
    # Positive:
    # prediction larger than ground truth
    #
    # Negative:
    # prediction smaller than ground truth
    # --------------------------------------------------------

    relative_volume_difference = safe_divide(
        prediction_count - ground_truth_count,
        ground_truth_count,
    )

    return {
        "metrics": {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "relative_volume_difference": relative_volume_difference,
        },

        "voxel_counts": {
            "prediction": prediction_count,
            "ground_truth": ground_truth_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },

        "empty_masks": {
            "prediction": prediction_count == 0,
            "ground_truth": ground_truth_count == 0,
        },
    }


# ============================================================
# 3. PROCESS ONE PATIENT
# ============================================================

def evaluate_patient(
    patient_id,
    prediction_path,
    ground_truth_path
):

    print("\n")
    print("=" * 70)
    print(f"EVALUATING PATIENT: {patient_id}")
    print("=" * 70)

    # ========================================================
    # LOAD FILES
    # ========================================================

    print("\nLoading prediction and ground truth...")

    pred_nii = nib.load(prediction_path)
    gt_nii = nib.load(ground_truth_path)

    # ========================================================
    # 4. CHECK SPATIAL ALIGNMENT
    # ========================================================

    if len(pred_nii.shape) != 3:
        raise ValueError(
            f"Prediction must be 3D, got {pred_nii.shape}"
        )

    if len(gt_nii.shape) != 3:
        raise ValueError(
            f"Ground truth must be 3D, got {gt_nii.shape}"
        )

    if pred_nii.shape != gt_nii.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{pred_nii.shape} vs {gt_nii.shape}"
        )

    for name, nii in [
        ("Prediction", pred_nii),
        ("Ground truth", gt_nii),
    ]:

        if not np.isfinite(nii.affine).all():
            raise ValueError(
                f"{name} has an invalid affine."
            )

    if not np.allclose(
        pred_nii.affine,
        gt_nii.affine,
        rtol=1e-5,
        atol=1e-4,
    ):
        raise ValueError(
            "Prediction and ground truth have "
            "different spatial grids."
        )

    # ========================================================
    # 5. SPATIAL UNITS
    # ========================================================

    pred_unit = (
        pred_nii
        .header
        .get_xyzt_units()[0]
    )

    gt_unit = (
        gt_nii
        .header
        .get_xyzt_units()[0]
    )

    if pred_unit != gt_unit:
        raise ValueError(
            f"Spatial units differ: "
            f"{pred_unit} vs {gt_unit}"
        )

    # ========================================================
    # 6. LOAD ARRAYS
    # ========================================================

    prediction = pred_nii.get_fdata()
    ground_truth = gt_nii.get_fdata()

    # ========================================================
    # 7. BASIC ARRAY VALIDATION
    # ========================================================

    for name, array in [
        ("Prediction", prediction),
        ("Ground truth", ground_truth),
    ]:

        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} contains non-finite values."
            )

        if (
            np.any(array < 0)
            or not np.all(array == np.rint(array))
        ):
            raise ValueError(
                f"{name} must contain "
                "non-negative integer labels."
            )

    prediction = prediction.astype(np.uint8)
    ground_truth = ground_truth.astype(np.uint8)

    # ========================================================
    # 8. VALIDATE LABEL CODINGS
    # ========================================================

    # Our MONAI post-processing creates:
    #
    # 0 = background
    # 1 = necrotic / non-enhancing tumor core
    # 2 = edema
    # 4 = enhancing tumor

    prediction_allowed_labels = [0, 1, 2, 4]

    if not np.isin(
        prediction,
        prediction_allowed_labels
    ).all():

        raise ValueError(
            "Unexpected prediction labels: "
            f"{np.unique(prediction).tolist()}"
        )

    # MSD Task01 BrainTumour ground truth uses:
    #
    # 0 = background
    # 1 = necrotic / non-enhancing tumor core
    # 2 = edema
    # 3 = enhancing tumor
    #
    # IMPORTANT:
    # Ground truth label 3 corresponds to the enhancing
    # tumor region that our prediction stores as label 4.

    ground_truth_allowed_labels = [0, 1, 2, 3]

    if not np.isin(
        ground_truth,
        ground_truth_allowed_labels
    ).all():

        raise ValueError(
            "Unexpected ground-truth labels: "
            f"{np.unique(ground_truth).tolist()}"
        )

    print(
        "Prediction labels:",
        np.unique(prediction)
    )

    print(
        "Ground truth labels:",
        np.unique(ground_truth)
    )

    print("Spatial grid checks passed.")

    # ========================================================
    # 9. DEFINE COMPARABLE BRATS REGIONS
    # ========================================================
    #
    # IMPORTANT:
    #
    # Prediction and ground truth do NOT use exactly the same
    # raw label numbers.
    #
    # Therefore we convert BOTH into the same clinically
    # relevant binary BraTS regions before calculating metrics.
    #
    #
    # PREDICTION
    # ----------
    #
    # WT = labels 1 + 2 + 4
    # TC = labels 1 + 4
    # ET = label 4
    #
    #
    # MSD GROUND TRUTH
    # ----------------
    #
    # WT = labels 1 + 2 + 3
    # TC = labels 1 + 3
    # ET = label 3
    #
    # ========================================================

    prediction_regions = {

        # Whole Tumor
        "whole_tumor":
            np.isin(
                prediction,
                [1, 2, 4]
            ),

        # Tumor Core
        "tumor_core":
            np.isin(
                prediction,
                [1, 4]
            ),

        # Enhancing Tumor
        "enhancing_tumor":
            prediction == 4,
    }

    ground_truth_regions = {

        # Whole Tumor
        "whole_tumor":
            np.isin(
                ground_truth,
                [1, 2, 3]
            ),

        # Tumor Core
        "tumor_core":
            np.isin(
                ground_truth,
                [1, 3]
            ),

        # Enhancing Tumor
        "enhancing_tumor":
            ground_truth == 3,
    }

    # ========================================================
    # 10. PHYSICAL VOXEL SIZE
    # ========================================================

    unit_to_mm = {
        "mm": 1.0,
        "meter": 1000.0,
        "micron": 0.001,
    }

    voxel_volume_mm3 = None

    if pred_unit in unit_to_mm:

        voxel_volume_mm3 = float(
            abs(
                np.linalg.det(
                    pred_nii.affine[:3, :3]
                )
            )
            * unit_to_mm[pred_unit] ** 3
        )

        if (
            not np.isfinite(voxel_volume_mm3)
            or voxel_volume_mm3 <= 0
        ):
            raise ValueError(
                "Invalid voxel volume."
            )

    # ========================================================
    # 11. EVALUATE REGIONS
    # ========================================================

    region_results = {}

    for region_name in [
        "whole_tumor",
        "tumor_core",
        "enhancing_tumor",
    ]:

        print(
            f"\nEvaluating {region_name}..."
        )

        result = evaluate_mask(
            prediction_regions[region_name],
            ground_truth_regions[region_name],
        )

        # ----------------------------------------------------
        # Physical volumes
        # ----------------------------------------------------

        if voxel_volume_mm3 is not None:

            pred_count = (
                result["voxel_counts"]["prediction"]
            )

            gt_count = (
                result["voxel_counts"]["ground_truth"]
            )

            result["volumes_ml"] = {

                "prediction": round(
                    pred_count
                    * voxel_volume_mm3
                    / 1000,
                    4
                ),

                "ground_truth": round(
                    gt_count
                    * voxel_volume_mm3
                    / 1000,
                    4
                ),

                "signed_difference": round(
                    (
                        pred_count
                        - gt_count
                    )
                    * voxel_volume_mm3
                    / 1000,
                    4
                ),
            }

        else:
            result["volumes_ml"] = None

        region_results[
            region_name
        ] = result

    # ========================================================
    # 12. STRUCTURED RESULT
    # ========================================================

    results = {

        "patient_id": patient_id,

        "prediction_file":
            prediction_path.name,

        "ground_truth_file":
            ground_truth_path.name,

        "evaluation_scope":
            "entire_3d_volume",

        "spatial_grid_checks_passed":
            True,

        "prediction_labels": [
            int(value)
            for value
            in np.unique(prediction)
        ],

        "ground_truth_labels": [
            int(value)
            for value
            in np.unique(ground_truth)
        ],

        # Explicitly record the different raw label schemes
        "label_schemes": {

            "prediction": {
                "background": 0,
                "necrotic_non_enhancing": 1,
                "edema": 2,
                "enhancing_tumor": 4,
            },

            "ground_truth": {
                "background": 0,
                "necrotic_non_enhancing": 1,
                "edema": 2,
                "enhancing_tumor": 3,
            },
        },

        "region_definitions": {

            "whole_tumor": {
                "prediction_labels": [1, 2, 4],
                "ground_truth_labels": [1, 2, 3],
            },

            "tumor_core": {
                "prediction_labels": [1, 4],
                "ground_truth_labels": [1, 3],
            },

            "enhancing_tumor": {
                "prediction_labels": [4],
                "ground_truth_labels": [3],
            },
        },

        "regions":
            region_results,

        "notes": [

            "Prediction and ground truth use different raw label encodings.",

            "Metrics are calculated after converting both label maps into equivalent BraTS regions.",

            "Undefined metrics are stored as null.",

            "Metrics are calculated over the entire 3D volume.",

            "Whole tumor, tumor core and enhancing tumor are evaluated separately.",

            "Matching spatial grids do not independently verify patient identity.",

            "These results evaluate segmentation agreement against the supplied ground truth.",

            "Results from a small number of cases do not establish general model performance.",
        ],
    }

    # ========================================================
    # 13. SAVE RESULT
    # ========================================================

    EVALUATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        EVALUATIONS_DIR
        / f"{patient_id}_evaluation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    # ========================================================
    # 14. PRINT RESULT
    # ========================================================

    print(
        "\n========== EVALUATION =========="
    )

    for region_name, region_result in region_results.items():

        print(
            f"\n{region_name.upper()}"
        )

        metrics = (
            region_result["metrics"]
        )

        for name, value in metrics.items():

            formatted = (
                "undefined"
                if value is None
                else f"{value:.4f}"
            )

            print(
                f"  {name}: {formatted}"
            )

        volumes = (
            region_result["volumes_ml"]
        )

        if volumes is not None:

            print(
                f"  Prediction volume: "
                f"{volumes['prediction']:.3f} ml"
            )

            print(
                f"  Ground truth volume: "
                f"{volumes['ground_truth']:.3f} ml"
            )

    print(
        "\nResults saved to:"
    )

    print(output_path)

    return results


# ============================================================
# 15. MAIN - ALL PATIENTS
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("NEUROSCOPE SEGMENTATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

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

    EVALUATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find patients
    # --------------------------------------------------------

    patient_dirs = sorted(
        directory
        for directory in DATA_DIR.iterdir()
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
    # Status lists
    # --------------------------------------------------------

    successful = []
    skipped = []
    failed = []

    # ========================================================
    # PROCESS PATIENTS
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
                "prediction not found."
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
                "ground truth not found."
            )

            skipped.append(
                patient_id
            )

            continue

        # ----------------------------------------------------
        # Evaluate patient
        # ----------------------------------------------------

        try:

            evaluate_patient(
                patient_id=patient_id,
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
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
                    str(error),
            })

    # ========================================================
    # 16. SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETE")
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
            "\nSuccessfully evaluated:"
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
            "\nFailed:"
        )

        for item in failed:

            print(
                f"  ✗ "
                f"{item['patient_id']}: "
                f"{item['error']}"
            )

    print(
        "\nEvaluation files:"
    )

    print(
        EVALUATIONS_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
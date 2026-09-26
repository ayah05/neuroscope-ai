from pathlib import Path
import json

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

PREDICTION_PATH = ROOT / "outputs" / "brats_457_prediction.nii.gz"
GROUND_TRUTH_PATH = (
    ROOT / "ml" / "data" / "real_patient" / "ground_truth.nii.gz"
)
OUTPUT_PATH = ROOT / "outputs" / "brats_457_evaluation.json"


def safe_divide(numerator, denominator):
    # Undefined metrics become null in JSON.
    return float(numerator / denominator) if denominator else None


def main():
    print("\nLoading prediction and ground truth...")

    pred_nii = nib.load(PREDICTION_PATH)
    gt_nii = nib.load(GROUND_TRUTH_PATH)

    # --------------------------------------------------------
    # 1. Check spatial alignment
    # --------------------------------------------------------

    if len(pred_nii.shape) != 3 or len(gt_nii.shape) != 3:
        raise ValueError("Both segmentations must be 3D.")

    if pred_nii.shape != gt_nii.shape:
        raise ValueError(
            f"Shape mismatch: {pred_nii.shape} vs {gt_nii.shape}"
        )

    for name, nii in [("Prediction", pred_nii), ("Ground truth", gt_nii)]:
        if not np.isfinite(nii.affine).all():
            raise ValueError(f"{name} has an invalid affine.")

    if not np.allclose(
        pred_nii.affine,
        gt_nii.affine,
        rtol=1e-5,
        atol=1e-4,
    ):
        raise ValueError(
            "Prediction and ground truth have different spatial grids. "
            "Check their alignment before evaluation."
        )

    pred_unit = pred_nii.header.get_xyzt_units()[0]
    gt_unit = gt_nii.header.get_xyzt_units()[0]

    if pred_unit != gt_unit:
        raise ValueError(
            f"Spatial units differ: {pred_unit} vs {gt_unit}"
        )

    prediction = pred_nii.get_fdata()
    ground_truth = gt_nii.get_fdata()

    for name, array in [
        ("Prediction", prediction),
        ("Ground truth", ground_truth),
    ]:
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values.")

        if np.any(array < 0) or not np.all(array == np.rint(array)):
            raise ValueError(
                f"{name} must contain non-negative integer labels."
            )

    if not np.isin(prediction, [0, 1, 2, 4]).all():
        raise ValueError("Unexpected prediction labels.")

    print("Prediction labels:", np.unique(prediction))
    print("Ground truth labels:", np.unique(ground_truth))
    print("Spatial grid checks passed.")

    # --------------------------------------------------------
    # 2. Whole-tumor masks
    #
    # All positive labels represent tumor-associated regions.
    # No comparison of individual tissue classes is performed.
    # --------------------------------------------------------

    pred_mask = prediction > 0
    gt_mask = ground_truth > 0

    true_positive = int(np.count_nonzero(pred_mask & gt_mask))
    false_positive = int(np.count_nonzero(pred_mask & ~gt_mask))
    false_negative = int(np.count_nonzero(~pred_mask & gt_mask))

    pred_count = int(np.count_nonzero(pred_mask))
    gt_count = int(np.count_nonzero(gt_mask))

    # --------------------------------------------------------
    # 3. Metrics over the entire 3D volume
    # --------------------------------------------------------

    dice = safe_divide(
        2 * true_positive,
        pred_count + gt_count,
    )

    iou = safe_divide(
        true_positive,
        true_positive + false_positive + false_negative,
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    # Positive: prediction is larger than ground truth.
    # Negative: prediction is smaller.
    relative_volume_difference = safe_divide(
        pred_count - gt_count,
        gt_count,
    )

    # --------------------------------------------------------
    # 4. Physical volumes, if spatial units are known
    # --------------------------------------------------------

    unit_to_mm = {
        "mm": 1.0,
        "meter": 1000.0,
        "micron": 0.001,
    }

    volumes = None

    if pred_unit in unit_to_mm:
        voxel_volume_mm3 = float(
            abs(np.linalg.det(pred_nii.affine[:3, :3]))
            * unit_to_mm[pred_unit] ** 3
        )

        if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
            raise ValueError("Invalid voxel volume.")

        volumes = {
            "prediction_ml": pred_count * voxel_volume_mm3 / 1000,
            "ground_truth_ml": gt_count * voxel_volume_mm3 / 1000,
            "signed_difference_ml": (
                (pred_count - gt_count) * voxel_volume_mm3 / 1000
            ),
        }

    # --------------------------------------------------------
    # 5. Save results
    # --------------------------------------------------------

    results = {
        "prediction_file": PREDICTION_PATH.name,
        "ground_truth_file": GROUND_TRUTH_PATH.name,
        "region": "whole_tumor_including_edema",
        "evaluation_scope": "entire_3d_volume",
        "mask_rule": "label > 0 for both segmentations",
        "spatial_grid_checks_passed": True,
        "prediction_labels": [
            int(value) for value in np.unique(prediction)
        ],
        "ground_truth_labels": [
            int(value) for value in np.unique(ground_truth)
        ],
        "metrics": {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "relative_volume_difference": relative_volume_difference,
        },
        "voxel_counts": {
            "prediction": pred_count,
            "ground_truth": gt_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "volumes": volumes,
        "empty_masks": {
            "prediction": pred_count == 0,
            "ground_truth": gt_count == 0,
        },
        "notes": [
            "Undefined metrics are stored as null.",
            "Individual tissue classes have not been evaluated.",
            "Matching spatial grids do not verify patient identity.",
            "This single-case result does not establish general model accuracy.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)

    print("\n========== WHOLE-TUMOR EVALUATION ==========")

    for name, value in results["metrics"].items():
        formatted = "undefined" if value is None else f"{value:.4f}"
        print(f"{name}: {formatted}")

    if volumes is not None:
        print(f"\nPrediction:   {volumes['prediction_ml']:.3f} ml")
        print(f"Ground truth: {volumes['ground_truth_ml']:.3f} ml")

    print("\nResults saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
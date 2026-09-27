from pathlib import Path
import modal


# ============================================================
# CONFIG
# ============================================================

app = modal.App("neuroscope-monai")

project_root = Path(__file__).resolve().parent.parent

LOCAL_DATA_DIR = (
    project_root
    / "ml"
    / "data"
    / "real_patients"
)

REMOTE_DATA_DIR = "/app/data"


# ============================================================
# MODAL IMAGE
# ============================================================

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "monai",
        "nibabel",
        "numpy",
        "pytorch-ignite"
    )
    .add_local_file(
        project_root
        / "bundles"
        / "brats_mri_segmentation"
        / "models"
        / "model.pt",
        "/app/model.pt"
    )
    .add_local_dir(
        LOCAL_DATA_DIR,
        REMOTE_DATA_DIR
    )
)


# ============================================================
# SHARED MODEL LOGIC
# used by run_inference() (batch) and segment() (backend upload)
# ============================================================

def load_model(device):
    """
    SegResNet exactly as in the bundle's inference.json,
    with the pretrained checkpoint loaded.
    """

    import torch

    from monai.networks.nets import SegResNet


    print("\n========== MODEL ==========")

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2
    ).to(device)


    checkpoint = torch.load(
        "/app/model.pt",
        map_location=device,
        weights_only=True
    )

    if "model" in checkpoint:
        model.load_state_dict(
            checkpoint["model"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print(
        "Model loaded on:",
        next(
            model.parameters()
        ).device
    )

    return model


def segment_volume(model, normalizer, image_np, device):
    """
    Input:  numpy [4, H, W, D], channel order T1ce, T1, T2, FLAIR
    Output: numpy [H, W, D] uint8 in BraTS labels
            (0 background, 1 necrotic/non-enhancing, 2 edema,
             4 enhancing)
    """

    import torch
    import numpy as np

    from monai.inferers import sliding_window_inference


    # ========================================================
    # NORMALIZATION
    # ========================================================

    image_tensor = torch.from_numpy(
        image_np
    )


    image_tensor = normalizer(
        image_tensor
    )


    # [4,H,W,D]
    # ->
    # [1,4,H,W,D]

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    print(
        "Model input:",
        image_tensor.shape
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    print(
        "\nRunning MONAI inference..."
    )


    with torch.no_grad():

        prediction = (
            sliding_window_inference(
                inputs=image_tensor,
                roi_size=(
                    240,
                    240,
                    160
                ),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )
        )


        prediction = torch.sigmoid(
            prediction
        )


    print(
        "Raw prediction:",
        prediction.shape
    )


    # ========================================================
    # THRESHOLD
    # ========================================================

    prediction = (
        prediction[0]
        > 0.5
    )


    # ========================================================
    # BRATS LABEL MAP
    #
    # Bundle inference.json:
    #
    # channel 2 -> label 4
    # channel 0 -> label 1
    # channel 1 -> label 2
    # ========================================================

    segmentation = torch.where(
        prediction[2],
        4,
        torch.where(
            prediction[0],
            1,
            torch.where(
                prediction[1],
                2,
                0
            )
        )
    )


    segmentation = (
        segmentation
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        "\nSegmentation:"
    )


    unique, counts = np.unique(
        segmentation,
        return_counts=True
    )


    for label, count in zip(
        unique,
        counts
    ):

        print(
            f"Label {label}: "
            f"{count} voxels"
        )


    # Free GPU memory before the next scan
    del image_tensor
    del prediction

    torch.cuda.empty_cache()


    return segmentation


# ============================================================
# GPU INFERENCE (BATCH: all patients in ml/data/real_patients)
# ============================================================

@app.function(
    image=image,
    gpu="T4",
    timeout=1200
)
def run_inference():

    from pathlib import Path

    import torch
    import nibabel as nib
    import numpy as np

    from monai.transforms import NormalizeIntensity


    # ========================================================
    # 1. GPU
    # ========================================================

    device = torch.device("cuda")

    print("\n========== GPU ==========")
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


    # ========================================================
    # 2. LOAD MODEL ONCE
    # ========================================================

    model = load_model(device)


    # ========================================================
    # 3. NORMALIZER
    # ========================================================

    normalizer = NormalizeIntensity(
        nonzero=True,
        channel_wise=True
    )


    # ========================================================
    # 4. FIND PATIENTS
    # ========================================================

    data_dir = Path(
        REMOTE_DATA_DIR
    )

    patient_dirs = sorted(
        [
            directory
            for directory
            in data_dir.iterdir()
            if directory.is_dir()
        ]
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
    # 5. RESULTS
    #
    # We'll return:
    #
    # {
    #     "BRATS_001": bytes,
    #     "BRATS_002": bytes,
    #     ...
    # }
    # ========================================================

    results = {}


    # ========================================================
    # 6. PROCESS EVERY PATIENT
    # ========================================================

    for index, patient_dir in enumerate(
        patient_dirs,
        start=1
    ):

        patient_id = patient_dir.name

        print("\n")
        print("=" * 70)

        print(
            f"PATIENT {index}/"
            f"{len(patient_dirs)}: "
            f"{patient_id}"
        )

        print("=" * 70)


        # ----------------------------------------------------
        # MRI paths
        #
        # IMPORTANT:
        # MONAI bundle expects:
        #
        # T1ce, T1, T2, FLAIR
        # ----------------------------------------------------

        paths = [
            patient_dir / "t1ce.nii.gz",
            patient_dir / "t1.nii.gz",
            patient_dir / "t2.nii.gz",
            patient_dir / "flair.nii.gz",
        ]


        # ----------------------------------------------------
        # Check required files
        # ----------------------------------------------------

        missing_files = [
            path.name
            for path in paths
            if not path.exists()
        ]


        if missing_files:

            print(
                f"[SKIP] Missing files: "
                f"{missing_files}"
            )

            continue


        # ====================================================
        # 7. LOAD MRI
        # ====================================================

        print(
            "\nLoading MRI modalities..."
        )


        volumes = []


        for path in paths:

            img = nib.load(
                str(path)
            )

            volume = (
                img
                .get_fdata()
                .astype(np.float32)
            )

            volumes.append(
                volume
            )

            print(
                f"{path.name}: "
                f"shape={volume.shape}, "
                f"min={volume.min():.2f}, "
                f"max={volume.max():.2f}"
            )


        # ----------------------------------------------------
        # [4, H, W, D]
        # ----------------------------------------------------

        image_np = np.stack(
            volumes,
            axis=0
        )


        print(
            "Stacked MRI:",
            image_np.shape
        )


        # ====================================================
        # 8.-12. NORMALIZATION, INFERENCE, LABEL MAP
        # ====================================================

        segmentation = segment_volume(
            model,
            normalizer,
            image_np,
            device
        )


        # ====================================================
        # 13. CREATE NIFTI
        # ====================================================

        reference = nib.load(
            str(paths[0])
        )


        result = nib.Nifti1Image(
            segmentation,
            reference.affine,
            reference.header
        )


        output_file = (
            f"/tmp/"
            f"{patient_id}_prediction.nii.gz"
        )


        nib.save(
            result,
            output_file
        )


        print(
            "Saved on Modal:",
            output_file
        )


        # ====================================================
        # 14. RETURN AS BYTES
        # ====================================================

        with open(
            output_file,
            "rb"
        ) as file:

            results[patient_id] = (
                file.read()
            )


        # GPU memory is freed inside segment_volume()


    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ALL PATIENTS FINISHED")
    print("=" * 70)

    print(
        "Predictions created:",
        len(results)
    )


    return results


# ============================================================
# GPU INFERENCE (SINGLE SCAN, called by backend/app.py)
# ============================================================

@app.function(
    image=image,
    gpu="T4",
    timeout=600
)
def segment(scan_bytes: bytes) -> bytes:
    """
    Input:  4D NIfTI (.nii.gz) as bytes, MSD format [H, W, D, 4],
            channels FLAIR, T1, T1ce, T2
    Output: segmentation as .nii.gz bytes in BraTS labels
    """

    import torch
    import nibabel as nib
    import numpy as np

    from monai.transforms import NormalizeIntensity


    device = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


    input_file = "/tmp/upload.nii.gz"

    with open(input_file, "wb") as file:
        file.write(scan_bytes)


    img = nib.load(input_file)

    data = img.get_fdata().astype(np.float32)

    print("Uploaded MRI shape:", data.shape)


    # MSD order (FLAIR, T1, T1ce, T2) -> model order
    # (T1ce, T1, T2, FLAIR), [H,W,D,4] -> [4,H,W,D]
    image_np = np.stack(
        [data[..., 2], data[..., 1], data[..., 3], data[..., 0]],
        axis=0
    )


    model = load_model(device)

    normalizer = NormalizeIntensity(
        nonzero=True,
        channel_wise=True
    )

    segmentation = segment_volume(
        model,
        normalizer,
        image_np,
        device
    )


    # 3D result: take affine and spatial unit from the input
    # (the 4D header itself does not fit a 3D volume).
    # tumor_features.py refuses volumes without a known unit.
    result = nib.Nifti1Image(segmentation, img.affine)

    spatial_unit, _ = img.header.get_xyzt_units()

    result.header.set_xyzt_units(
        spatial_unit if spatial_unit != "unknown" else "mm"
    )


    output_file = "/tmp/prediction.nii.gz"

    nib.save(result, output_file)

    with open(output_file, "rb") as file:
        return file.read()


# ============================================================
# LOCAL
# ============================================================

@app.local_entrypoint()
def main():

    print(
        "Starting MONAI batch GPU inference..."
    )


    # --------------------------------------------------------
    # Run remotely
    # --------------------------------------------------------

    results = (
        run_inference.remote()
    )


    # --------------------------------------------------------
    # Local output directory
    # --------------------------------------------------------

    output_dir = (
        project_root
        / "outputs"
        / "predictions"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save every prediction
    # --------------------------------------------------------

    for (
        patient_id,
        prediction_bytes
    ) in results.items():


        output_file = (
            output_dir
            / f"{patient_id}_prediction.nii.gz"
        )


        output_file.write_bytes(
            prediction_bytes
        )


        print(
            f"[SAVED] "
            f"{patient_id} -> "
            f"{output_file}"
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "BATCH INFERENCE COMPLETE"
    )

    print("=" * 70)

    print(
        f"Predictions saved: "
        f"{len(results)}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )
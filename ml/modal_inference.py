from pathlib import Path
import modal


app = modal.App("neuroscope-monai")

project_root = Path(__file__).resolve().parent.parent

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
        project_root / "bundles/brats_mri_segmentation/models/model.pt",
        "/app/model.pt"
    )
    .add_local_dir(
        Path(__file__).parent / "data/real_patient",
        "/app/data"
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600
)
def run_inference():

    import torch
    import nibabel as nib
    import numpy as np

    from monai.networks.nets import SegResNet
    from monai.transforms import NormalizeIntensity
    from monai.inferers import sliding_window_inference

    # --------------------------------------------------
    # 1. GPU
    # --------------------------------------------------

    device = torch.device("cuda")

    print("\n========== GPU ==========")
    print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # 2. MRI DATEIEN LADEN
    # WICHTIG: Reihenfolge T1ce, T1, T2, FLAIR
    # --------------------------------------------------

    paths = [
        "/app/data/patient_t1ce.nii.gz",
        "/app/data/patient_t1.nii.gz",
        "/app/data/patient_t2.nii.gz",
        "/app/data/patient_flair.nii.gz",
    ]

    volumes = []

    for path in paths:
        img = nib.load(path)

        volume = img.get_fdata().astype(np.float32)

        volumes.append(volume)

        print(
            path,
            "shape:",
            volume.shape,
            "min:",
            volume.min(),
            "max:",
            volume.max()
        )

    # [4, H, W, D]
    image_np = np.stack(volumes, axis=0)

    print("\nStacked MRI shape:", image_np.shape)

    # --------------------------------------------------
    # 3. NORMALISIERUNG
    # gleiche Idee wie im MONAI Bundle
    # --------------------------------------------------

    image_tensor = torch.tensor(image_np)

    normalizer = NormalizeIntensity(
        nonzero=True,
        channel_wise=True
    )

    image_tensor = normalizer(image_tensor)

    # Batch dimension hinzufügen:
    # [4,H,W,D] -> [1,4,H,W,D]

    image_tensor = image_tensor.unsqueeze(0).to(device)

    print("Model input shape:", image_tensor.shape)
    print("Input device:", image_tensor.device)

    # --------------------------------------------------
    # 4. SEGRESNET ERSTELLEN
    # exakt entsprechend inference.json
    # --------------------------------------------------

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2
    ).to(device)

    # --------------------------------------------------
    # 5. PRETRAINED CHECKPOINT LADEN
    # --------------------------------------------------

    checkpoint = torch.load(
        "/app/model.pt",
        map_location=device,
        weights_only=True
    )

    print("\nCheckpoint loaded.")

    # MONAI checkpoint enthält normalerweise "model"
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded on:", next(model.parameters()).device)

    # --------------------------------------------------
    # 6. INFERENCE
    # --------------------------------------------------

    print("\n========== INFERENCE ==========")

    with torch.no_grad():

        prediction = sliding_window_inference(
            inputs=image_tensor,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5
        )

        prediction = torch.sigmoid(prediction)

    print("Raw prediction shape:", prediction.shape)

    # --------------------------------------------------
    # 7. THRESHOLD
    # --------------------------------------------------

    prediction = prediction[0] > 0.5

    print("Binary prediction shape:", prediction.shape)

    # --------------------------------------------------
    # 8. BRAts LABEL MAP
    #
    # exakt entsprechend der Bundle inference.json:
    #
    # channel 2 -> label 4
    # channel 0 -> label 1
    # channel 1 -> label 2
    # --------------------------------------------------

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

    segmentation = segmentation.cpu().numpy().astype(np.uint8)

    print("\n========== RESULT ==========")

    unique, counts = np.unique(
        segmentation,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(f"Label {label}: {count} voxels")

    # --------------------------------------------------
    # 9. NIFTI ERZEUGEN
    # --------------------------------------------------

    reference = nib.load(paths[0])

    result = nib.Nifti1Image(
        segmentation,
        reference.affine,
        reference.header
    )

    # This path exists inside the Modal container
    output_file = "/tmp/brats_457_prediction.nii.gz"

    nib.save(result, output_file)

    print("\nSaved on Modal:", output_file)

    with open(output_file, "rb") as f:
        return f.read()


# --------------------------------------------------
# LOKALER TEIL
# --------------------------------------------------
@app.local_entrypoint()
def main():

    print("Starting MONAI GPU inference...")

    # Runs in Modal cloud
    result = run_inference.remote()

    # This runs locally on your PC
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "brats_457_prediction.nii.gz"

    output_file.write_bytes(result)

    print("\n================================")
    print("DONE!")
    print("Segmentation saved locally to:")
    print(output_file)
    print("================================")
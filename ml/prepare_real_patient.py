from pathlib import Path
import tarfile
import nibabel as nib


ROOT = Path(__file__).parent.parent

TAR_FILE = ROOT / "downloads" / "Task01_BrainTumour.tar"
OUTPUT_DIR = ROOT / "ml" / "data" / "real_patient"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. TAR öffnen und Inhalt untersuchen
# --------------------------------------------------

print("Opening archive...")

with tarfile.open(TAR_FILE, "r") as tar:

    members = tar.getmembers()

    image_files = [
        m for m in members
        if "imagesTr/" in m.name and m.name.endswith(".nii.gz")
    ]

    label_files = [
        m for m in members
        if "labelsTr/" in m.name and m.name.endswith(".nii.gz")
    ]

    print(f"Training images found: {len(image_files)}")
    print(f"Training labels found: {len(label_files)}")

    if not image_files:
        raise RuntimeError("No imagesTr NIfTI files found!")

    # Für unsere Demo einfach den ersten Patienten nehmen
    patient_image = image_files[0]

    patient_filename = Path(patient_image.name).name

    print("\nSelected patient:")
    print(patient_filename)

    # Passendes Ground-Truth-Label suchen
    patient_label = next(
        (
            m for m in label_files
            if Path(m.name).name == patient_filename
        ),
        None
    )

    if patient_label is None:
        raise RuntimeError(
            f"No matching label found for {patient_filename}"
        )

    # --------------------------------------------------
    # 2. Nur diesen Patienten extrahieren
    # --------------------------------------------------

    print("\nExtracting MRI...")

    image_source = tar.extractfile(patient_image)

    temp_image = OUTPUT_DIR / "patient_4d.nii.gz"

    with open(temp_image, "wb") as f:
        f.write(image_source.read())

    print("Extracting ground truth...")

    label_source = tar.extractfile(patient_label)

    ground_truth = OUTPUT_DIR / "ground_truth.nii.gz"

    with open(ground_truth, "wb") as f:
        f.write(label_source.read())


# --------------------------------------------------
# 3. 4D MRI laden
# --------------------------------------------------

print("\nLoading MRI...")

img = nib.load(temp_image)

data = img.get_fdata()

print("MRI shape:", data.shape)


if len(data.shape) != 4 or data.shape[3] != 4:
    raise RuntimeError(
        f"Expected 4 MRI channels, got shape {data.shape}"
    )


# --------------------------------------------------
# 4. MSD BrainTumour Modalitäten
#
# Channel 0 = FLAIR
# Channel 1 = T1
# Channel 2 = T1ce
# Channel 3 = T2
# --------------------------------------------------

modalities = {
    "flair": data[..., 0],
    "t1": data[..., 1],
    "t1ce": data[..., 2],
    "t2": data[..., 3],
}


# --------------------------------------------------
# 5. Als einzelne NIfTI-Dateien speichern
# --------------------------------------------------

for name, volume in modalities.items():

    output_file = OUTPUT_DIR / f"patient_{name}.nii.gz"

    output_img = nib.Nifti1Image(
        volume,
        img.affine,
        img.header
    )

    nib.save(output_img, output_file)

    print(f"Saved {name}: {output_file}")


# temporäre 4D-Datei entfernen
temp_image.unlink()


print("\n===================================")
print("REAL PATIENT READY")
print("===================================")

print("\nFiles:")

for file in OUTPUT_DIR.iterdir():
    print(" -", file.name)
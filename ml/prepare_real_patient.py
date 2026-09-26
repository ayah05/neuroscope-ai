from pathlib import Path
import tarfile
import requests
import nibabel as nib
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).parent.parent

TAR_FILE = ROOT / "downloads/Task01_BrainTumour.tar"

OUTPUT_DIR = (
    ROOT
    / "ml"
    / "data"
    / "real_patients"
)

NUMBER_OF_PATIENTS = 5


# Medical Segmentation Decathlon
DOWNLOAD_URL = (
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/"
    "Task01_BrainTumour.tar"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DOWNLOAD DATASET
# ============================================================

def download_dataset():

    # --------------------------------------------------------
    # Dataset already exists
    # --------------------------------------------------------

    if TAR_FILE.exists():

        size_gb = (
            TAR_FILE.stat().st_size
            / 1024 ** 3
        )

        print(
            f"Dataset already downloaded: "
            f"{TAR_FILE}"
        )

        print(
            f"Size: {size_gb:.2f} GB"
        )

        return


    # --------------------------------------------------------
    # Download dataset
    # --------------------------------------------------------

    print("\nDataset not found.")
    print("Downloading MSD BrainTumour dataset...\n")

    response = requests.get(
        DOWNLOAD_URL,
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    chunk_size = (
        1024 * 1024
    )  # 1 MB


    with open(
        TAR_FILE,
        "wb"
    ) as file:

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Task01_BrainTumour.tar"
        ) as progress:

            for chunk in response.iter_content(
                chunk_size=chunk_size
            ):

                if not chunk:
                    continue

                file.write(chunk)

                progress.update(
                    len(chunk)
                )


    print(
        "\nDataset download complete."
    )


# ============================================================
# CHECK IF PATIENT IS ALREADY PREPARED
# ============================================================

def patient_is_ready(
    patient_dir
):

    required_files = [

        patient_dir
        / "flair.nii.gz",

        patient_dir
        / "t1.nii.gz",

        patient_dir
        / "t1ce.nii.gz",

        patient_dir
        / "t2.nii.gz",

        patient_dir
        / "ground_truth.nii.gz",
    ]


    return all(
        file.exists()
        for file in required_files
    )


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(
    tar,
    patient_image,
    label_files
):

    patient_filename = Path(
        patient_image.name
    ).name


    patient_id = (
        patient_filename
        .replace(
            ".nii.gz",
            ""
        )
    )


    patient_dir = (
        OUTPUT_DIR
        / patient_id
    )


    # --------------------------------------------------------
    # Patient already prepared
    # --------------------------------------------------------

    if patient_is_ready(
        patient_dir
    ):

        print(
            f"[SKIP] "
            f"{patient_id} "
            f"already prepared."
        )

        return


    print("\n")
    print("=" * 60)

    print(
        f"Processing patient: "
        f"{patient_id}"
    )

    print("=" * 60)


    patient_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Find matching ground truth
    # --------------------------------------------------------

    patient_label = next(

        (
            m
            for m in label_files

            if Path(
                m.name
            ).name
            == patient_filename
        ),

        None
    )


    if patient_label is None:

        print(
            f"[WARNING] "
            f"No label found "
            f"for {patient_id}"
        )

        return


    # --------------------------------------------------------
    # Extract 4D MRI
    # --------------------------------------------------------

    print(
        "Extracting MRI..."
    )


    image_source = (
        tar.extractfile(
            patient_image
        )
    )


    temp_image = (
        patient_dir
        / "patient_4d.nii.gz"
    )


    with open(
        temp_image,
        "wb"
    ) as file:

        file.write(
            image_source.read()
        )


    # --------------------------------------------------------
    # Extract ground truth
    # --------------------------------------------------------

    print(
        "Extracting ground truth..."
    )


    label_source = (
        tar.extractfile(
            patient_label
        )
    )


    ground_truth = (
        patient_dir
        / "ground_truth.nii.gz"
    )


    with open(
        ground_truth,
        "wb"
    ) as file:

        file.write(
            label_source.read()
        )


    # --------------------------------------------------------
    # Load MRI
    # --------------------------------------------------------

    print(
        "Loading MRI..."
    )


    img = nib.load(
        temp_image
    )


    data = (
        img.get_fdata()
    )


    print(
        "MRI shape:",
        data.shape
    )


    if (
        len(data.shape) != 4
        or data.shape[3] != 4
    ):

        raise RuntimeError(

            f"Expected 4 MRI channels, "
            f"got {data.shape} "
            f"for {patient_id}"
        )


    # ========================================================
    # MSD BrainTumour Modalities
    #
    # Channel 0 = FLAIR
    # Channel 1 = T1
    # Channel 2 = T1ce
    # Channel 3 = T2
    # ========================================================

    modalities = {

        "flair":
            data[..., 0],

        "t1":
            data[..., 1],

        "t1ce":
            data[..., 2],

        "t2":
            data[..., 3],
    }


    # --------------------------------------------------------
    # Save modalities
    # --------------------------------------------------------

    for (
        name,
        volume
    ) in modalities.items():


        output_file = (
            patient_dir
            / f"{name}.nii.gz"
        )


        output_img = (
            nib.Nifti1Image(
                volume,
                img.affine,
                img.header
            )
        )


        nib.save(
            output_img,
            output_file
        )


        print(
            f"Saved {name}: "
            f"{output_file.name}"
        )


    # --------------------------------------------------------
    # Remove temporary file
    # --------------------------------------------------------

    if temp_image.exists():

        temp_image.unlink()


    print(
        f"[DONE] {patient_id}"
    )


# ============================================================
# PREPARE PATIENTS
# ============================================================

def prepare_patients():

    print(
        "\nOpening TAR archive..."
    )


    with tarfile.open(
        TAR_FILE,
        "r"
    ) as tar:


        members = (
            tar.getmembers()
        )

        image_files = [
            member
            for member in members
            if (
                    "imagesTr/" in member.name
                    and member.name.endswith(".nii.gz")
                    and not Path(member.name).name.startswith("._")
            )
        ]

        label_files = [
            member
            for member in members
            if (
                    "labelsTr/" in member.name
                    and member.name.endswith(".nii.gz")
                    and not Path(member.name).name.startswith("._")
            )
        ]


        print(
            f"Training images found: "
            f"{len(image_files)}"
        )


        print(
            f"Training labels found: "
            f"{len(label_files)}"
        )


        if not image_files:

            raise RuntimeError(
                "No training images found."
            )


        # ----------------------------------------------------
        # Sort patients
        # ----------------------------------------------------

        image_files = sorted(
            image_files,
            key=lambda x: x.name
        )


        # ----------------------------------------------------
        # Select first N patients
        # ----------------------------------------------------

        selected_patients = (
            image_files[
                :NUMBER_OF_PATIENTS
            ]
        )


        print(
            f"\nPreparing "
            f"{len(selected_patients)} "
            f"patients:\n"
        )


        for patient in selected_patients:

            print(
                " -",
                Path(
                    patient.name
                ).name
            )


        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        print()


        for patient_image in selected_patients:

            process_patient(

                tar=tar,

                patient_image=patient_image,

                label_files=label_files
            )


# ============================================================
# SUMMARY
# ============================================================

def print_summary():

    print("\n")
    print("=" * 60)

    print(
        "DATASET PREPARATION COMPLETE"
    )

    print("=" * 60)


    patient_dirs = sorted(

        directory

        for directory
        in OUTPUT_DIR.iterdir()

        if directory.is_dir()
    )


    print(
        f"\nPrepared patients: "
        f"{len(patient_dirs)}"
    )


    for patient_dir in patient_dirs:

        status = (

            "READY"

            if patient_is_ready(
                patient_dir
            )

            else "INCOMPLETE"
        )


        print(
            f" - {patient_dir.name}: "
            f"{status}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("NEUROSCOPE DATASET PREPARATION")
    print("=" * 60)


    # --------------------------------------------------------
    # Step 1
    # Download dataset if necessary
    # --------------------------------------------------------

    download_dataset()


    # --------------------------------------------------------
    # Step 2
    # Prepare patients
    # --------------------------------------------------------

    prepare_patients()


    # --------------------------------------------------------
    # Step 3
    # Summary
    # --------------------------------------------------------

    print_summary()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
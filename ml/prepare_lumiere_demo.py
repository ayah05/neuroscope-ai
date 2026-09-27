from pathlib import Path
import zipfile
import json
import shutil


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

LUMIERE_DIR = ROOT / "ml" / "data" / "lumiere"

ZIP_FILE = LUMIERE_DIR / "Imaging-v202211.zip"

OUTPUT_DIR = LUMIERE_DIR / "demo"

# Start with this patient because we already know from inspection
# that it has multiple longitudinal timepoints.
PATIENT_ID = "Patient-064"


# ============================================================
# FILES WE WANT FROM EACH TIMEPOINT
# ============================================================

FILES_TO_EXTRACT = {
    "T1.nii.gz": "t1.nii.gz",
    "CT1.nii.gz": "t1ce.nii.gz",
    "T2.nii.gz": "t2.nii.gz",
    "FLAIR.nii.gz": "flair.nii.gz",

    "DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz":
        "segmentation.nii.gz",

    "DeepBraTumIA-segmentation/atlas/segmentation/measured_volumes_in_mm3.json":
        "measured_volumes_in_mm3.json",
}


# ============================================================
# FIND TIMEPOINTS
# ============================================================

def find_timepoints(archive):

    prefix = f"Imaging/{PATIENT_ID}/"

    timepoints = set()

    for name in archive.namelist():

        if not name.startswith(prefix):
            continue

        relative = name[len(prefix):]

        parts = relative.split("/")

        if not parts:
            continue

        week = parts[0]

        if week.startswith("week-"):
            timepoints.add(week)

    return sorted(
        timepoints,
        key=lambda x: int(x.split("-")[1])
    )


# ============================================================
# EXTRACT ONE FILE
# ============================================================

def extract_file(
    archive,
    archive_path,
    output_path
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with archive.open(archive_path) as src:
        with open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


# ============================================================
# PROCESS TIMEPOINT
# ============================================================

def process_timepoint(
    archive,
    week
):

    print(f"\nProcessing {week}")

    archive_files = set(
        archive.namelist()
    )

    output_week = (
        OUTPUT_DIR
        / PATIENT_ID
        / week
    )

    extracted = {}

    for source_name, output_name in FILES_TO_EXTRACT.items():

        archive_path = (
            f"Imaging/"
            f"{PATIENT_ID}/"
            f"{week}/"
            f"{source_name}"
        )

        if archive_path not in archive_files:

            print(
                f"  [MISSING] "
                f"{source_name}"
            )

            extracted[output_name] = False

            continue

        output_path = (
            output_week
            / output_name
        )

        print(
            f"  Extracting "
            f"{output_name}"
        )

        extract_file(
            archive,
            archive_path,
            output_path
        )

        extracted[output_name] = True

    return extracted


# ============================================================
# CREATE SYNTHETIC PATIENT CONTEXT
# ============================================================

def create_patient_context():

    context = {

        "patient_id": PATIENT_ID,

        "demo_metadata": {
            "clinical_context_source": "synthetic",
            "imaging_source": "LUMIERE"
        },

        "demographics": {
            "display_name": "Maria Beispiel",
            "age": 55,
            "sex": "Female"
        },

        "clinical_context": {

            "chief_complaint":
                "Progressive headaches over 6 weeks, "
                "new word-finding difficulties",

            "symptoms": [
                "Morning headaches, worsening",
                "Word-finding difficulties",
                "Focal seizure involving right arm"
            ],

            "conditions": [
                "Arterial hypertension",
                "Hypothyroidism"
            ],

            "medications": [
                "Ramipril 5 mg",
                "Levothyroxine 75 µg",
                "Levetiracetam 500 mg"
            ],

            "allergies": [
                "Penicillin"
            ]
        },

        "oncology_context": {
            "diagnosis": "Glioblastoma",
            "treatment_status": "Follow-up",

            # Keep treatment details unknown unless we
            # deliberately define them for the demo.
            "prior_surgery": None,
            "radiotherapy": None,
            "chemotherapy": None,
            "steroids": None,
            "mgmt_status": None,
            "idh_status": None
        }
    }

    patient_dir = (
        OUTPUT_DIR
        / PATIENT_ID
    )

    patient_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        patient_dir
        / "patient_context.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            context,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nCreated patient context:\n"
        f"{output_file}"
    )


# ============================================================
# CREATE MANIFEST
# ============================================================

def create_manifest(results):

    manifest = {
        "patient_id": PATIENT_ID,
        "timepoints": []
    }

    for week, files in results.items():

        week_number = int(
            week.split("-")[1]
        )

        manifest["timepoints"].append({
            "week": week_number,
            "directory": week,
            "files": files
        })

    output_file = (
        OUTPUT_DIR
        / PATIENT_ID
        / "manifest.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )

    print(
        f"\nCreated manifest:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not ZIP_FILE.exists():

        raise FileNotFoundError(
            f"LUMIERE archive not found:\n"
            f"{ZIP_FILE}"
        )

    print("=" * 70)
    print("PREPARING LUMIERE DEMO PATIENT")
    print("=" * 70)

    print(
        f"\nPatient: {PATIENT_ID}"
    )

    with zipfile.ZipFile(
        ZIP_FILE,
        "r"
    ) as archive:

        timepoints = find_timepoints(
            archive
        )

        if not timepoints:

            raise RuntimeError(
                f"No timepoints found for "
                f"{PATIENT_ID}"
            )

        print(
            f"\nFound {len(timepoints)} "
            f"timepoints:"
        )

        for week in timepoints:
            print(f"  {week}")

        results = {}

        for week in timepoints:

            results[week] = (
                process_timepoint(
                    archive,
                    week
                )
            )

    create_patient_context()

    create_manifest(
        results
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nDemo patient saved to:\n"
        f"{OUTPUT_DIR / PATIENT_ID}"
    )


if __name__ == "__main__":
    main()
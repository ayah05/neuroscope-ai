from pathlib import Path

import nibabel as nib
import numpy as np


output_dir = Path("data/test_patient")
output_dir.mkdir(parents=True, exist_ok=True)

# Etwas kleiner als echtes BraTS.
# SlidingWindowInferer kann mit kleineren Volumes umgehen/padden.
shape = (128, 128, 128)

# Identische räumliche Orientierung für alle Modalitäten
affine = np.eye(4)

rng = np.random.default_rng(42)

modalities = {
    "t1ce": rng.normal(100, 20, shape),
    "t1": rng.normal(90, 20, shape),
    "t2": rng.normal(110, 25, shape),
    "flair": rng.normal(120, 25, shape),
}

for name, volume in modalities.items():

    volume = volume.astype(np.float32)

    image = nib.Nifti1Image(
        volume,
        affine
    )

    path = output_dir / f"test_{name}.nii.gz"

    nib.save(image, path)

    print(f"Saved: {path}")

print("Finished.")
from pathlib import Path

import requests
from tqdm import tqdm


URL = (
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/"
    "Task01_BrainTumour.tar"
)

OUTPUT_DIR = Path("downloads")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "Task01_BrainTumour.tar"


def download_file(url, output_path):
    print("Connecting to server...")

    response = requests.get(
        url,
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get("content-length", 0)
    )

    print(f"File: {output_path.name}")

    if total_size:
        print(
            f"Size: {total_size / 1024**3:.2f} GB"
        )

    print("\nStarting download...\n")

    with open(output_path, "wb") as file:

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="BrainTumour",
        ) as progress:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(chunk)

                    progress.update(len(chunk))

    print("\nDownload complete!")
    print("Saved to:")
    print(output_path.resolve())


if __name__ == "__main__":
    download_file(URL, OUTPUT_FILE)
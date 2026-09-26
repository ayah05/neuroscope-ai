from pathlib import Path
import requests
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DOWNLOAD_DIR = Path("data/lumiere")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Figshare article containing the MRI data + segmentations
FIGSHARE_API_URL = (
    "https://api.figshare.com/v2/articles/21249516"
)


# ============================================================
# GET DOWNLOAD INFORMATION FROM FIGSHARE
# ============================================================

def get_figshare_files():
    """
    Fetch file metadata from the official Figshare API.
    """

    print("Fetching LUMIERE file information...")

    response = requests.get(
        FIGSHARE_API_URL,
        timeout=30
    )

    response.raise_for_status()

    article = response.json()

    files = article.get("files", [])

    if not files:
        raise RuntimeError(
            "No downloadable files were found on Figshare."
        )

    return files


# ============================================================
# DOWNLOAD FILE
# ============================================================

def download_file(url, output_path):

    print(f"\nDownloading: {output_path.name}")

    # Resume download if a partial file already exists
    existing_size = 0

    if output_path.exists():
        existing_size = output_path.stat().st_size

    headers = {}

    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

        print(
            f"Resuming existing download at "
            f"{existing_size / 1024**3:.2f} GB"
        )

    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    # If server ignores Range, restart download
    if existing_size > 0 and response.status_code == 200:
        print(
            "Server did not accept resume request. "
            "Restarting download."
        )

        existing_size = 0
        mode = "wb"

    else:
        mode = "ab" if existing_size > 0 else "wb"

    content_length = int(
        response.headers.get("content-length", 0)
    )

    total_size = existing_size + content_length

    chunk_size = 1024 * 1024  # 1 MB

    with open(output_path, mode) as file:

        with tqdm(
            total=total_size,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=output_path.name
        ) as progress:

            for chunk in response.iter_content(
                chunk_size=chunk_size
            ):

                if not chunk:
                    continue

                file.write(chunk)
                progress.update(len(chunk))

    print(f"\nSaved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    files = get_figshare_files()

    print("\nAvailable files:\n")

    for index, file_info in enumerate(files):

        name = file_info["name"]
        size = file_info.get("size", 0)

        print(
            f"[{index}] {name} "
            f"({size / 1024**3:.2f} GB)"
        )

    print()

    # Download every file belonging to this Figshare article
    for file_info in files:

        filename = file_info["name"]
        download_url = file_info["download_url"]

        output_path = DOWNLOAD_DIR / filename

        download_file(
            download_url,
            output_path
        )


if __name__ == "__main__":
    main()
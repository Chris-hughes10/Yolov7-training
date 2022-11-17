### This script can be run from the root folder of the original Yolov7 Repo to extract the state dicts


import os
import subprocess
from pathlib import Path

import requests
import torch


def attempt_download(file, repo="WongKinYiu/yolov7"):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", "").lower())

    if not file.exists():
        # Set default assets
        assets = [
            "yolov7.pt",
            "yolov7-tiny.pt",
            "yolov7x.pt",
            "yolov7-d6.pt",
            "yolov7-e6.pt",
            "yolov7-e6e.pt",
            "yolov7-w6.pt",
        ]
        try:
            response = requests.get(
                f"https://api.github.com/repos/{repo}/releases"
            ).json()  # github api
            if len(response) > 0:
                release_assets = response[0]  # get dictionary of assets
                # Get names of assets if it rleases exists
                assets = [
                    release_asset["name"] for release_asset in release_assets["assets"]
                ]
                # Get first tag which is the latest tag
                tag = release_assets.get("tag_name")
        except KeyError:  # fallback plan
            tag = subprocess.check_output("git tag", shell=True).decode().split()[-1]
        except subprocess.CalledProcessError:  # fallback to default release if can't get tag
            tag = "v0.1"

        name = file.name
        if name in assets:
            msg = f"{file} missing, try downloading from https://github.com/{repo}/releases/"
            redundant = False  # second download option
            try:  # GitHub
                url = f"https://github.com/{repo}/releases/download/{tag}/{name}"
                print(f"Downloading {url} to {file}...")
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1e6  # check
            except Exception as e:  # GCP
                print(f"Download error: {e}")
                assert redundant, "No secondary mirror"
                url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
                print(f"Downloading {url} to {file}...")
                # torch.hub.download_url_to_file(url, weights)
                os.system(f"curl -L {url} -o {file}")
            finally:
                if not file.exists() or file.stat().st_size < 1e6:  # check
                    file.unlink(missing_ok=True)  # remove partial downloads
                    print(f"ERROR: Download failure: {msg}")
                print("")
                return


def save_state_dict(weights, output_path):
    # with distributed zero first, local_process_zero_first decorator from pt acc
    attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location="cpu")  # load checkpoint

    loaded_model = ckpt["model"]
    torch.save(loaded_model.float().state_dict(), output_path)


if __name__ == "__main__":
    checkpoints = [
        "yolov7_training",
        "yolov7x_training",
        "yolov7-d6_training",
        "yolov7-e6_training",
        "yolov7-e6e_training",
        "yolov7-w6_training",
    ]

    for weights in checkpoints:
        model = save_state_dict(f"{weights}.pt", f"{weights}_state_dict.pt")

    print("done")

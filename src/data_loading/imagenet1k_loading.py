import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_unzip_imagenet1k(kaggle_key_path: str, download_path: str):
    """
    Downloads the ImageNet1K dataset from Kaggle and unzips it to the specified location.
    """
    # Kaggle API credentials, pull w/ api
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(kaggle_key_path)
    api = KaggleApi()
    api.authenticate()

    # Download & unzip
    print("Downloading ImageNet1K dataset...")
    api.dataset_download_files('vitaliykinakh/stable-imagenet1k', path=download_path, unzip=False)
    print("Download complete. Unzipping...")
    zip_file_path = os.path.join(download_path, 'stable-imagenet1k.zip')
    if os.path.exists(zip_file_path):
        print("Unzipping ImageNet1K dataset...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(zip_file_path)
        print("Dataset downloaded and unzipped successfully.")
    else:
        print("Error: Zip file not found. Please check the download path.")

if __name__ == "__main__":
    kaggle_json_path = "~/.kaggle/"
    download_path = "/content/drive/MyDrive/EmbarkLabs/"
    download_and_unzip_imagenet1k(kaggle_json_path, download_path)

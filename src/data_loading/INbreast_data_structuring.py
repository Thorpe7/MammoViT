import os
import shutil
import pandas as pd

def load_csv(csv_path):
    """Load the CSV file and return a DataFrame."""
    return pd.read_csv(csv_path, sep=';')

def simplify_birads(birads):
    """Simplify Bi-Rads scores to their base integer values.
    (e.g., '4a' -> 4)
    """
    if isinstance(birads, str):
        return int(birads[0])
    return int(birads)

def organize_files_by_birads(csv_path, dicom_dir, output_dir):
    """
    Organize DICOM files into folders based on their Bi-Rads scores.

    Args:
        csv_path (str): Path to the CSV file.
        dicom_dir (str): Path to the directory containing DICOM files.
        output_dir (str): Path to the output directory for organized files.
    """
    # Load the CSV file
    df = load_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over df to organize files
    for _, row in df.iterrows():
        file_id = str(row['File Name'])  # Ensure file_id is a string
        birads = simplify_birads(row['Bi-Rads'])

        # Create the Bi-Rads folder if it doesn't exist
        birads_folder = os.path.join(output_dir, f"BIRADS_{birads}")
        os.makedirs(birads_folder, exist_ok=True)

        # Search for the file containing the unique ID
        found = False
        for dicom_file in os.listdir(dicom_dir):
            if file_id in dicom_file:
                src_path = os.path.join(dicom_dir, dicom_file)
                dest_path = os.path.join(birads_folder, dicom_file)
                shutil.move(src_path, dest_path)
                found = True
                break

        if not found:
            print(f"File not found for ID: {file_id}")


if __name__ == "__main__":
    # Local Use
    # organize_files_by_birads(
    #     csv_path='/Users/thorpe/git_repos/MammoViT/data/INbreast/INbreast.csv',
    #     dicom_dir='/Users/thorpe/git_repos/MammoViT/data/INbreast/AllDICOMs',
    #     output_dir='/Users/thorpe/git_repos/MammoViT/data/INbreast/OrganizedByBiRads'
    # )
    # Colab Use
    organize_files_by_birads(
        csv_path='/content/drive/MyDrive/EmbarkLabs/INbreast_Release_1.0/INbreast.csv',
        dicom_dir='/content/drive/MyDrive/EmbarkLabs/INbreast_Release_1.0/AllDICOMs',
        output_dir='/content/drive/MyDrive/EmbarkLabs/INbreast_Release_1.0/OrganizedByBiRads'
    )

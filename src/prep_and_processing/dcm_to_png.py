import os
from pathlib import Path
import pydicom
from PIL import Image
import numpy as np

def convert_dicom_to_png_dir(
    src_root: str = "OrganizedByBiRads",
    dst_root: str = "OrganizedByBiRads_PNG"
):
    src_root = Path(src_root) # type: ignore
    dst_root = Path(dst_root) # type: ignore
    dst_root.mkdir(parents=True, exist_ok=True) # type: ignore

    for class_dir in src_root.iterdir(): # type: ignore
        if not class_dir.is_dir():
            continue

        dst_class_dir = dst_root / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for dcm_file in class_dir.glob("*.dcm"):
            try:
                dicom = pydicom.dcmread(dcm_file)
                pixel_array = dicom.pixel_array

                # Normalize to 0-255 and convert to uint8
                pixel_array = pixel_array.astype(np.float32)
                pixel_array -= pixel_array.min()
                pixel_array /= (pixel_array.max() + 1e-5)
                pixel_array *= 255.0
                pixel_array = pixel_array.astype(np.uint8)

                img = Image.fromarray(pixel_array).convert("L")  # grayscale
                png_filename = dcm_file.stem + ".png"
                img.save(dst_class_dir / png_filename)

            except Exception as e:
                print(f"Failed to convert {dcm_file}: {e}")

    print(f"All DICOMs converted to PNG in: {dst_root}")
if __name__ == "__main__":
    # Local run
    convert_dicom_to_png_dir(
        src_root="/Users/thorpe/git_repos/MammoViT/data/INbreast/OrganizedByBiRads",
        dst_root="/Users/thorpe/git_repos/MammoViT/data/INbreast/OrganizedByBiRads_PNG"
)
    # Use w/ colab
#     convert_dicom_to_png_dir(
#         src_root="/Users/thorpe/git_repos/MammoViT/data/INbreast/OrganizedByBiRads",
#         dst_root="/Users/thorpe/git_repos/MammoViT/data/INbreast/OrganizedByBiRads_PNG"
# )
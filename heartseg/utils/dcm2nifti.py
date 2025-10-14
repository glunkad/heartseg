import os
import glob
import subprocess
import shutil
import argparse


def convert_dcm2nifti(dicom_dir, output_dir):
    """
    Converts DICOM files in a directory to a single merged NIfTI file using dcm2niix.
    """
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    cmd = ["dcm2niix", "-m", "y", "-z", "y", "-f", "%p", "-o", temp_dir, dicom_dir]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        nifti_files = glob.glob(os.path.join(temp_dir, "*.nii.gz"))
        if not nifti_files:
            print("No NIfTI file created.")
            return

        main_nifti = max(nifti_files, key=os.path.getsize)
        final_path = os.path.join(output_dir, f"{os.path.basename(dicom_dir)}.nii.gz")
        shutil.move(main_nifti, final_path)
        print(f"Saved: {final_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.strip()}")

    shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Convert a DICOM folder to NIfTI (.nii.gz).")
    parser.add_argument("dicom_dir", help="Path to the DICOM directory.")
    parser.add_argument("output_dir", help="Path to the output directory.")
    args = parser.parse_args()

    convert_dcm2nifti(args.dicom_dir, args.output_dir)


if __name__ == "__main__":
    main()
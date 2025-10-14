import shutil
import os
import argparse
from heartseg.utils.dcm2nifti import convert_dcm2nifti


def preprocess_data(input_dir, output_dir, temp_dir=None):
    """
    Converts all DICOM subjects in `input_dir` into NIfTI files compatible with nnU-Net.

    Args:
        input_dir (str): Path to input directory containing DICOM subject folders.
        output_dir (str): Path to save NIfTI files.
        temp_dir (str, optional): Temporary directory used during conversion.
    """
    if temp_dir is None:
        temp_dir = os.path.join(output_dir, "temp_conversion")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    for subject_name in sorted(os.listdir(input_dir)):
        subject_dicom_dir = os.path.join(input_dir, subject_name)

        if os.path.isdir(subject_dicom_dir):
            print(f"Processing subject: {subject_name}")
            subject_temp_dir = os.path.join(temp_dir, subject_name)
            os.makedirs(subject_temp_dir, exist_ok=True)

            # 1. Convert DICOM → NIfTI
            converted_nifti_path = convert_dcm2nifti(subject_dicom_dir, subject_temp_dir)

            if converted_nifti_path:
                # 2. Rename and move to final directory
                final_nifti_name = f"{subject_name}_0000.nii.gz"
                final_nifti_path = os.path.join(output_dir, final_nifti_name)
                shutil.move(converted_nifti_path, final_nifti_path)
                print(f"  -> Saved to: {final_nifti_path}\n")

    # Cleanup temp folder
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print("Temporary conversion directory cleaned up.")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DICOM files into single NIfTI files for nnU-Net inference."
    )
    parser.add_argument("--input_root", required=True, help="Root directory with subject DICOM folders.")
    parser.add_argument("--output_root", required=True, help="Output directory for converted NIfTI files.")
    args = parser.parse_args()

    preprocess_data(args.input_root, args.output_root)


if __name__ == "__main__":
    main()

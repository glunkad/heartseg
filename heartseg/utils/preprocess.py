import shutil
import os
import argparse
from heartseg.utils.dcm2nifti import convert_dcm2nifti

def preprocess_data(temp_dir, input, output):
    for subject_name in sorted(os.listdir(input)):
        subject_dicom_dir = os.path.join(input, subject_name)

        if os.path.isdir(subject_dicom_dir):
            print(f"Processing subject: {subject_name}")
            subject_temp_dir = os.path.join(temp_dir, subject_name)

            # 1. Convert DICOM to a single NIfTI file in a temporary location.
            converted_nifti_path = convert_dcm2nifti(subject_dicom_dir, subject_temp_dir)

            if converted_nifti_path:
                # 2. Rename and move the file to the final nnU-Net input directory.
                final_nifti_name = f"{subject_name}_0000.nii.gz"
                final_nifti_path = os.path.join(output, final_nifti_name)
                shutil.move(converted_nifti_path, final_nifti_path)
                print(f"  -> Saved to: {final_nifti_path}\n")

        # Clean up the temporary directory.
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print("Temporary conversion directory cleaned up.")

def main():
    """
    Parses arguments and processes a root directory of DICOMs for nnU-Net inference.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess DICOM files into single NIfTI files for nnU-Net inference."
    )
    parser.add_argument(
        "--input_root",
        required=True,
        help="Path to the root directory containing subject folders (e.g., 'data/dicoms/')."
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Path to the final output directory for nnU-Net compatible NIfTI files (e.g., 'data/nnunet_input/')."
    )

    args = parser.parse_args()

    # A temporary directory to handle intermediate files from dcm2niix.
    # This keeps the final output directory clean.
    input_root = args.input_root
    output_root = args.output_root

    temp_dir = os.path.join(output_root, "temp_conversion")
    preprocess_data(temp_dir, input_root, output_root)

    # Ensure the final output directory exists and is clean.
    os.makedirs(output_root, exist_ok=True)


if __name__ == "__main__":
    main()
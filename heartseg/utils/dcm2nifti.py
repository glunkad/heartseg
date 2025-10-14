import subprocess
import os
import argparse
import glob


def convert_dcm2nifti(dicom_dir, temp_output_dir):
    """
    Converts a DICOM series to a single, merged NIfTI file using dcm2niix.

    Args:
        dicom_dir (str): Path to the directory containing a DICOM series.
        temp_output_dir (str): A temporary directory to store the conversion output.

    Returns:
        str: The path to the primary generated NIfTI file, or None if conversion fails.
    """
    os.makedirs(temp_output_dir, exist_ok=True)

    # Key dcm2niix flags for nnU-Net:
    # -m y: Merge slices from the same series into a single 3D/4D file. CRITICAL for nnU-Net.
    # -z y: Compress the output NIfTI to .nii.gz.
    # -f %p: Use the protocol name for the filename (a simple, predictable starting point).
    command = [
        "dcm2niix",
        "-m", "y",
        "-z", "y",
        "-f", "%p",
        "-o", temp_output_dir,
        dicom_dir
    ]

    try:
        # Run the command, hiding the verbose output on success for a cleaner log.
        subprocess.run(command, check=True, capture_output=True, text=True)

        # --- Find the correct output file ---
        # dcm2niix might create multiple files (e.g., main image, phase image).
        # We need to find the primary volume.
        nifti_files = glob.glob(os.path.join(temp_output_dir, "*.nii.gz"))

        if not nifti_files:
            print("  -> FAILED: No NIfTI file was created.")
            return None

        # A common heuristic: the largest file is usually the main image volume.
        # This avoids picking smaller, secondary images like phase maps.
        main_nifti_file = max(nifti_files, key=os.path.getsize)
        print(f"  -> Successfully converted. Primary file found: {os.path.basename(main_nifti_file)}")
        return main_nifti_file

    except subprocess.CalledProcessError as e:
        print(f"  -> FAILED: dcm2niix returned an error for directory: {dicom_dir}")
        print(f"  -> Error: {e.stderr.strip()}")
        return None


def main():
    """
    Parses arguments and processes a root directory of DICOMs for nnU-Net inference.
    """
    parser = argparse.ArgumentParser(
        description="Convert DICOM directories into single NIfTI files for nnU-Net inference."
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
    temp_dir = os.path.join(args.output_root, "temp_conversion")

    # Ensure the final output directory exists and is clean.
    os.makedirs(args.output_root, exist_ok=True)

    # Loop through each subject folder in the input directory.
    for subject_name in sorted(os.listdir(args.input_root)):
        subject_dicom_dir = os.path.join(args.input_root, subject_name)

        if os.path.isdir(subject_dicom_dir):
            print(f"Processing subject: {subject_name}")

            # Use a subject-specific temporary folder.
            subject_temp_dir = os.path.join(temp_dir, subject_name)

            # 1. Convert DICOM to a single NIfTI file in a temporary location.
            converted_nifti_path = convert_dcm2nifti(subject_dicom_dir, subject_temp_dir)

            if converted_nifti_path:
                # 2. Rename and move the file to the final nnU-Net input directory.
                final_nifti_name = f"{subject_name}_0000.nii.gz"
                final_nifti_path = os.path.join(args.output_root, final_nifti_name)

                # Use os.rename for an efficient move operation.
                os.rename(converted_nifti_path, final_nifti_path)
                print(f"  -> Saved to: {final_nifti_path}\n")

    # Clean up the temporary directory.
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
        print("Temporary conversion directory cleaned up.")


if __name__ == "__main__":
    main()
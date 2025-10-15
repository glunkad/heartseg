import argparse
import subprocess
import os


def convert_nifti2dcmseg(nifti_path, dicom_dir, json_path, output_dicom_path):
    """
    Converts a NIfTI segmentation to DICOM SEG using the dcmqi command-line tool.

    Args:
        nifti_path (str): Path to the input NIfTI segmentation file (.nii or .nii.gz).
        dicom_dir (str): Path to the directory containing the original DICOM series.
        json_path (str): Path to the JSON metadata file describing the segments.
        output_dicom_path (str): Path for the output DICOM SEG file (.dcm).
    """
    command = [
        "C:\\Softwares\\dcmqi-1.3.5\\bin\\itkimage2segimage",
        "--inputImageList", nifti_path,
        "--inputMetadata", json_path,
        "--inputDICOMDirectory", dicom_dir,
        "--outputDICOM", output_dicom_path,
    ]

    try:
        print(f"Executing command: {' '.join(command)}")
        subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Conversion successful! Output saved to: {output_dicom_path}")
    except subprocess.CalledProcessError as e:
        print("Conversion failed.")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert NIfTI files to DICOM seg format."
    )
    parser.add_argument(
        "--nifti_file",
        required=True,
        help="Path to the segmentation file (.nii or .nii.gz).",
    )

    parser.add_argument(
        "--dicom_ref_dir",
        required=True,
        help="Path to the dicom reference dir",
    )

    parser.add_argument(
        "--json",
        required=True,
        help="Path to the json file",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output seg file file",
    )

    args = parser.parse_args()

    NIFTI_FILE = args.nifti_file
    DICOM_REF_DIR = args.dicom_ref_dir
    JSON_META_FILE = args.json
    OUTPUT_SEG_DICOM = args.output

    # Ensure all input files exist
    if not all(os.path.exists(p) for p in [NIFTI_FILE, DICOM_REF_DIR, JSON_META_FILE]):
        print("Error: One or more input files/directories not found.")
    else:
        convert_nifti2dcmseg(NIFTI_FILE, DICOM_REF_DIR, JSON_META_FILE, OUTPUT_SEG_DICOM)


if __name__ == "__main__":
    main()

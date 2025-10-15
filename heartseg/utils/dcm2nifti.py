import os
import argparse
import SimpleITK as sitk


def convert_dcm2nifti(dicom_dir, output_dir):
    """
    Converts DICOM files in a directory to a single merged NIfTI file using SimpleITK.
    """

    os.makedirs(output_dir, exist_ok=True)
    nifti_path = os.path.join(output_dir, f"{os.path.basename(dicom_dir)}.nii.gz")

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()
    sitk.WriteImage(image, nifti_path)

    print(f"Saved: {nifti_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert a DICOM folder to NIfTI (.nii.gz).")
    parser.add_argument("--dicom_dir", help="Path to the DICOM directory.")
    parser.add_argument("--output_dir", help="Path to the output directory.")

    args = parser.parse_args()

    if args.dicom_dir and args.output_dir:
        convert_single_dicom_series(args.dicom_dir, args.output_dir)


if __name__ == "__main__":
    main()

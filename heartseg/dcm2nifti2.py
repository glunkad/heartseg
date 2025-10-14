import os
import argparse
import SimpleITK as sitk

def convert_single_dicom_series(dicom_series_folder, output_nifti_path):
    dicom_series_folder = os.path.abspath(dicom_series_folder)
    output_nifti_path = os.path.abspath(output_nifti_path)

    os.makedirs(os.path.dirname(output_nifti_path), exist_ok=True)

    dicom_files = [os.path.join(dicom_series_folder, f)
                   for f in os.listdir(dicom_series_folder)
                   if f.lower().endswith(('.dcm', '.dic'))]

    if not dicom_files:
        raise ValueError(f"No DICOM (.dcm) files found in: {dicom_series_folder}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted(dicom_files))
    image = reader.Execute()

    sitk.WriteImage(image, output_nifti_path)
    print(f"NIfTI Converted and saved: {output_nifti_path}")


def main():
    parser = argparse.ArgumentParser(description="DICOM <-> NIfTI conversion script using SimpleITK")
    parser.add_argument("--dicom_dir", help="Path to DICOM series folder")
    parser.add_argument("--nifti_out", help="Output NIfTI path")

    args = parser.parse_args()

    if args.dicom_dir and args.nifti_out:
        convert_single_dicom_series(args.dicom_dir, args.nifti_out)


if __name__ == "__main__":
    main()

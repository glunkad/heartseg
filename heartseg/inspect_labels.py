import nibabel as nib
import numpy as np
import sys

# Check if a file path is provided
if len(sys.argv) < 2:
    print("Usage: python inspect_labels.py /path/to/your/segmentation.nii.gz")
    sys.exit(1)

# Get the file path from the command-line argument
nifti_file_path = sys.argv[1]

try:
    # Load the NIfTI image
    image = nib.load(nifti_file_path)

    # Get the image data as a NumPy array
    data = image.get_fdata()

    # Find all unique integer values in the data
    unique_labels = np.unique(data)

    # Convert them to integers for a clean list
    unique_labels = [int(label) for label in unique_labels]

    print(f"Found the following unique labels in '{nifti_file_path}':")
    print(sorted(unique_labels))

except Exception as e:
    print(f"An error occurred: {e}")
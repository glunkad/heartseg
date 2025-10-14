
# Here is an
# example
# of
# the
# required
# JSON
# metadata
# file
# structure:
# ```json
# {
#     "segments": [
#         {
#             "label_id": 1,
#             "label_name": "Right Ventricle",
#             "rgb_color": [255, 0, 0],
#             "anatomic_region": {
#                 "code_value": "80891009",
#                 "scheme_designator": "SCT",
#                 "code_meaning": "Right ventricle"
#             }
#         },
#         {
#             "label_id": 2,
#             "label_name": "Myocardium",
#             "rgb_color": [0, 255, 0],
#             "anatomic_region": {
#                 "code_value": "74220003",
#                 "scheme_designator": "SCT",
#                 "code_meaning": "Myocardium"
#             }
#         }
#     ]
# }
# ```

import os
import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from datetime import datetime
import logging
from skimage import color
import warnings
import json
import argparse
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NiftiToDicomSeg:
    def __init__(self, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.segments = metadata.get("segments", [])
        if not self.segments:
            raise ValueError("Metadata JSON must contain a non-empty 'segments' list.")
        logger.info(f"Loaded {len(self.segments)} segment definitions from metadata.")

    def load_dicom_series(self, dicom_dir):
        """Load DICOM series from directory"""
        dicom_files = []
        for filename in sorted(os.listdir(dicom_dir)):
            filepath = os.path.join(dicom_dir, filename)
            try:
                ds = pydicom.dcmread(filepath)
                if hasattr(ds, 'pixel_array'):
                    dicom_files.append(ds)
            except Exception as e:
                logger.warning(f"Could not read or is not an image file {filepath}: {e}")
                continue

        def sort_key(ds):
            if hasattr(ds, 'ImagePositionPatient'):
                return float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation'):
                return float(ds.SliceLocation)
            elif hasattr(ds, 'InstanceNumber'):
                return int(ds.InstanceNumber)
            else:
                return 0

        dicom_files.sort(key=sort_key)
        logger.info(f"Loaded {len(dicom_files)} DICOM image files.")
        return dicom_files

    def load_nifti_segmentation(self, nifti_path):
        """Load NIfTI segmentation file"""
        nifti_img = nib.load(nifti_path)
        seg_data = nifti_img.get_fdata().astype(np.uint8)
        logger.info(f"Loaded NIfTI segmentation with shape: {seg_data.shape}")
        return seg_data, nifti_img

    def rgb_to_cielab_dicom(self, rgb_8bit):
        """Converts an 8-bit RGB color to a DICOM-compliant CIELab value."""
        rgb_float = np.array(rgb_8bit, dtype=np.uint8).reshape(1, 1, 3)
        lab_float = color.rgb2lab(rgb_float)
        L_star, a_star, b_star = lab_float[0][0]
        L_dicom = int((L_star / 100.0) * 65535)
        a_dicom = int(((a_star + 128.0) / 255.0) * 65535)
        b_dicom = int(((b_star + 128.0) / 255.0) * 65535)
        return [L_dicom, a_dicom, b_dicom]

    def create_segment_sequence(self):
        """Create segment sequence for DICOM SEG from metadata."""
        segment_sequence = []
        for i, seg_info in enumerate(self.segments):
            segment_item = Dataset()
            segment_item.SegmentNumber = i + 1
            segment_item.SegmentLabel = seg_info["label_name"]
            segment_item.SegmentAlgorithmType = "AUTOMATIC"
            segment_item.SegmentAlgorithmName = "AI Segmentation"
            segment_item.RecommendedDisplayCIELabValue = self.rgb_to_cielab_dicom(seg_info["rgb_color"])

            anatomic_region = Dataset()
            anat_info = seg_info["anatomic_region"]
            anatomic_region.CodeValue = anat_info["code_value"]
            anatomic_region.CodingSchemeDesignator = anat_info["scheme_designator"]
            anatomic_region.CodeMeaning = anat_info["code_meaning"]
            segment_item.AnatomicRegionSequence = [anatomic_region]

            property_category = Dataset()
            property_category.CodeValue = "123037004"
            property_category.CodingSchemeDesignator = "SCT"
            property_category.CodeMeaning = "Anatomical Structure"
            segment_item.SegmentedPropertyCategoryCodeSequence = [property_category]

            segment_sequence.append(segment_item)
        return segment_sequence

    def create_referenced_series_sequence(self, dicom_series):
        """Create referenced series sequence"""
        ref_series_item = Dataset()
        ref_series_item.SeriesInstanceUID = dicom_series[0].SeriesInstanceUID
        ref_instance_sequence = []
        for ds in dicom_series:
            ref_instance_item = Dataset()
            ref_instance_item.ReferencedSOPClassUID = ds.SOPClassUID
            ref_instance_item.ReferencedSOPInstanceUID = ds.SOPInstanceUID
            ref_instance_sequence.append(ref_instance_item)
        ref_series_item.ReferencedInstanceSequence = ref_instance_sequence
        return [ref_series_item]

    def convert_nifti_to_dicom_orientation(self, nifti_data):
        """
        Convert NIfTI to DICOM orientation.
        NOTE: This is a heuristic and may not work for all NIfTI orientations.
        NIfTI (RAS+) to DICOM (LPS+): Transpose X and Y.
        """
        return np.transpose(nifti_data, (1, 0, 2))

    def create_shared_functional_groups(self, reference_dicom):
        """Create shared functional groups sequence"""
        shared_fg = Dataset()
        pixel_measures = Dataset()
        pixel_measures.PixelSpacing = reference_dicom.PixelSpacing
        pixel_measures.SliceThickness = reference_dicom.SliceThickness
        shared_fg.PixelMeasuresSequence = [pixel_measures]

        plane_orientation = Dataset()
        plane_orientation.ImageOrientationPatient = reference_dicom.ImageOrientationPatient
        shared_fg.PlaneOrientationSequence = [plane_orientation]
        return [shared_fg]

    def create_per_frame_functional_groups(self, dicom_series):
        """Create per-frame functional groups sequence"""
        per_frame_functional_groups = []
        for i, ds in enumerate(dicom_series):
            frame_fg = Dataset()
            derivation_image_item = Dataset()
            source_image_item = Dataset()
            source_image_item.ReferencedSOPClassUID = ds.SOPClassUID
            source_image_item.ReferencedSOPInstanceUID = ds.SOPInstanceUID
            derivation_image_item.SourceImageSequence = [source_image_item]
            derivation_code = Dataset()
            derivation_code.CodeValue = "113076"
            derivation_code.CodingSchemeDesignator = "DCM"
            derivation_code.CodeMeaning = "Segmentation"
            derivation_image_item.DerivationCodeSequence = [derivation_code]
            frame_fg.DerivationImageSequence = [derivation_image_item]

            plane_pos = Dataset()
            plane_pos.ImagePositionPatient = ds.ImagePositionPatient
            frame_fg.PlanePositionSequence = [plane_pos]
            per_frame_functional_groups.append(frame_fg)
        return per_frame_functional_groups

    def convert_to_dicom_seg(self, nifti_path, dicom_dir, output_path):
        """Main conversion function"""
        dicom_series = self.load_dicom_series(dicom_dir)
        if not dicom_series:
            raise ValueError("No DICOM image files found in the specified directory.")

        seg_data, _ = self.load_nifti_segmentation(nifti_path)
        reference_dicom = dicom_series[0]

        # This orientation conversion is a common case but might need adjustment
        seg_data = self.convert_nifti_to_dicom_orientation(seg_data)

        if seg_data.shape != (reference_dicom.Rows, reference_dicom.Columns, len(dicom_series)):
            logger.warning(
                f"Shape mismatch between NIfTI ({seg_data.shape}) and DICOM series ({(reference_dicom.Rows, reference_dicom.Columns, len(dicom_series))}).")

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"  # Segmentation Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # Copy patient and study info
        ds.PatientName = reference_dicom.PatientName
        ds.PatientID = reference_dicom.PatientID
        ds.StudyInstanceUID = reference_dicom.StudyInstanceUID
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        now = datetime.now()
        ds.InstanceCreationDate = now.strftime('%Y%m%d')
        ds.InstanceCreationTime = now.strftime('%H%M%S.%f')
        ds.SeriesDate = now.strftime('%Y%m%d')
        ds.SeriesTime = now.strftime('%H%M%S.%f')
        ds.ContentDate = now.strftime('%Y%m%d')
        ds.ContentTime = now.strftime('%H%M%S.%f')

        ds.Modality = "SEG"
        ds.SeriesDescription = "AI Segmentation"
        ds.SeriesNumber = "999"
        ds.ImageType = ["DERIVED", "PRIMARY"]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = reference_dicom.Rows, reference_dicom.Columns
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
        ds.PixelRepresentation = 0
        ds.SegmentationType = "BINARY"

        ds.SegmentSequence = self.create_segment_sequence()
        ds.ReferencedSeriesSequence = self.create_referenced_series_sequence(dicom_series)
        ds.SharedFunctionalGroupsSequence = self.create_shared_functional_groups(reference_dicom)

        # Create concatenated pixel data for all segments
        all_frames_data = []
        for seg_info in self.segments:
            label_id = seg_info["label_id"]
            binary_mask = (seg_data == label_id).astype(np.uint8)
            for i in range(len(dicom_series)):
                frame = binary_mask[:, :, i] if i < binary_mask.shape[2] else np.zeros((ds.Rows, ds.Columns),
                                                                                       dtype=np.uint8)
                all_frames_data.append(frame.tobytes())

        ds.PixelData = b''.join(all_frames_data)
        ds.NumberOfFrames = len(all_frames_data)

        # Per-frame functional groups
        per_frame_fg_sequence = []
        for i in range(len(self.segments)):
            for j in range(len(dicom_series)):
                frame_fg = Dataset()
                frame_fg.SegmentIdentificationSequence = [Dataset()]
                frame_fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber = i + 1

                plane_pos = Dataset()
                plane_pos.ImagePositionPatient = dicom_series[j].ImagePositionPatient
                frame_fg.PlanePositionSequence = [plane_pos]

                derivation_image_item = Dataset()
                source_image_item = Dataset()
                source_image_item.ReferencedSOPClassUID = dicom_series[j].SOPClassUID
                source_image_item.ReferencedSOPInstanceUID = dicom_series[j].SOPInstanceUID
                derivation_image_item.SourceImageSequence = [source_image_item]
                frame_fg.DerivationImageSequence = [derivation_image_item]

                per_frame_fg_sequence.append(frame_fg)
        ds.PerFrameFunctionalGroupsSequence = per_frame_fg_sequence

        ds.save_as(output_path, write_like_original=False)
        logger.info(f"Successfully created DICOM SEG file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a NIfTI segmentation to a DICOM SEG object.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "nifti_path",
        type=Path,
        help="Path to the input NIfTI segmentation file (.nii or .nii.gz)."
    )
    parser.add_argument(
        "dicom_dir",
        type=Path,
        help="Path to the directory containing the original DICOM series."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the JSON metadata file describing the segments."
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path for the output DICOM SEG file (.dcm)."
    )
    args = parser.parse_args()

    if not args.nifti_path.exists():
        logger.error(f"NIfTI file not found: {args.nifti_path}")
        return
    if not args.dicom_dir.is_dir():
        logger.error(f"DICOM directory not found: {args.dicom_dir}")
        return
    if not args.json_path.exists():
        logger.error(f"JSON metadata file not found: {args.json_path}")
        return

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        converter = NiftiToDicomSeg(metadata_path=args.json_path)
        converter.convert_to_dicom_seg(
            nifti_path=args.nifti_path,
            dicom_dir=args.dicom_dir,
            output_path=args.output_path
        )
        print(f" DICOM SEG conversion successful. Output saved to: {args.output_path}")
    except Exception as e:
        logger.error(f"An error occurred during conversion: {e}", exc_info=True)
        print(f"!Error during conversion: {e}")


if __name__ == "__main__":
    main()
import os
import json
import argparse
import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from datetime import datetime
import logging
from skimage import color
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class NiftiToDicomSegConverter:
    """
    A generalized class to convert a NIfTI segmentation mask into a DICOM SEG object
    using a JSON configuration file for metadata.
    """

    def __init__(self, config_path):
        """
        Initializes the converter by loading metadata from a JSON config file.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load or parse the configuration JSON: {e}")

        # Parse the config into a format the script can use
        self.label_mapping = {int(k): v["description"] for k, v in self.config["segments"].items()}
        self.sct_codes = {int(k): v["sctCodes"] for k, v in self.config["segments"].items()}
        self.color_map = {v["description"]: v["color"] for k, v in self.config["segments"].items()}
        logger.info(f"Configuration loaded for '{self.config.get('seriesDescription', 'Unknown Series')}'")

    def load_dicom_series(self, dicom_dir):
        """Loads and sorts a DICOM series from a directory."""
        dicom_files = []
        for filename in sorted(os.listdir(dicom_dir)):
            if filename.lower().endswith(('.dcm', '.ima', '.dic')) or '.' not in filename:
                filepath = os.path.join(dicom_dir, filename)
                try:
                    ds = pydicom.dcmread(filepath)
                    dicom_files.append(ds)
                except Exception:
                    continue

        dicom_files.sort(key=lambda ds: float(ds.ImagePositionPatient[2]) if 'ImagePositionPatient' in ds else int(
            ds.InstanceNumber))
        logger.info(f"Loaded {len(dicom_files)} DICOM files from '{dicom_dir}'")
        return dicom_files

    def load_nifti_segmentation(self, nifti_path):
        """Loads a NIfTI segmentation mask."""
        nifti_img = nib.load(nifti_path)
        seg_data = nifti_img.get_fdata().astype(np.uint8)
        logger.info(f"Loaded NIfTI segmentation '{os.path.basename(nifti_path)}' with shape: {seg_data.shape}")
        return seg_data

    def create_segment_sequence(self, unique_labels):
        """Creates the SegmentSequence for the DICOM SEG object based on the loaded config."""
        segment_sequence = []

        def rgb_to_cielab_dicom(rgb_8bit):
            rgb_float = np.array(rgb_8bit, dtype=np.uint8).reshape(1, 1, 3)
            lab_float = color.rgb2lab(rgb_float)
            L, a, b = lab_float[0][0]
            return [int((L / 100.0) * 65535), int(((a + 128.0) / 255.0) * 65535), int(((b + 128.0) / 255.0) * 65535)]

        for label in sorted(unique_labels):
            if label == 0: continue

            label_int = int(label)
            name = self.label_mapping.get(label_int)
            codes = self.sct_codes.get(label_int)

            if not name or not codes:
                logger.warning(f"Label {label_int} found in NIfTI but is not defined in the JSON config. Skipping.")
                continue

            segment_item = Dataset()
            segment_item.SegmentNumber = len(segment_sequence) + 1
            segment_item.SegmentLabel = name
            segment_item.SegmentAlgorithmType = "AUTOMATIC"
            segment_item.SegmentAlgorithmName = self.config.get("algorithmName", "AI Segmentation")

            assigned_color = self.color_map.get(name, [128, 128, 128])
            segment_item.RecommendedDisplayCIELabValue = rgb_to_cielab_dicom(assigned_color)

            # Add required anatomical codes from config
            segment_item.SegmentedPropertyCategoryCodeSequence = [Dataset().from_json(json.dumps(codes["category"]))]
            segment_item.SegmentedPropertyTypeCodeSequence = [Dataset().from_json(json.dumps(codes["type"]))]
            if "anatomicRegion" in codes:
                segment_item.AnatomicRegionSequence = [Dataset().from_json(json.dumps(codes["anatomicRegion"]))]

            segment_sequence.append(segment_item)

        return segment_sequence

    def convert_to_dicom_seg(self, nifti_path, dicom_dir, output_path):
        """Main conversion logic."""
        dicom_series = self.load_dicom_series(dicom_dir)
        if not dicom_series:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

        seg_data = self.load_nifti_segmentation(nifti_path)
        ref_dicom = dicom_series[0]

        unique_labels = [l for l in np.unique(seg_data) if l != 0]
        logger.info(f"Found unique labels in NIfTI mask: {unique_labels}")

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)

        ds.PatientName = ref_dicom.PatientName
        ds.PatientID = ref_dicom.PatientID
        ds.StudyInstanceUID = ref_dicom.StudyInstanceUID

        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = int(ref_dicom.SeriesNumber) + 100
        ds.SeriesDescription = self.config.get("seriesDescription", "Segmentation")
        ds.Modality = "SEG"
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = 1
        ds.BodyPartExamined = self.config.get("bodyPartExamined", "")

        now = datetime.now()
        ds.InstanceCreationDate = now.strftime('%Y%m%d')
        ds.InstanceCreationTime = now.strftime('%H%M%S.%f')
        ds.ContentDate = now.strftime('%Y%m%d')
        ds.ContentTime = now.strftime('%H%M%S.%f')

        ds.ImageType = ["DERIVED", "SECONDARY"]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = ref_dicom.Rows
        ds.Columns = ref_dicom.Columns
        ds.BitsAllocated = 8
        ds.BitsStored = 1
        ds.HighBit = 0
        ds.PixelRepresentation = 0

        ds.SegmentationType = "BINARY"
        ds.ContentLabel = "SEGMENTATION"
        ds.ContentCreatorName = self.config.get("contentCreatorName", "AI Model")

        ds.SegmentSequence = self.create_segment_sequence(unique_labels)
        if not ds.SegmentSequence:
            raise ValueError("No valid segments were created. Check if labels in NIfTI match the config JSON.")

        # --- Generic DICOM construction functions (largely unchanged) ---
        def create_ref_series_sequence(dicom_series):
            ref = Dataset()
            ref.SeriesInstanceUID = dicom_series[0].SeriesInstanceUID
            ref.ReferencedInstanceSequence = []
            for s in dicom_series:
                item = Dataset()
                item.ReferencedSOPClassUID = s.SOPClassUID
                item.ReferencedSOPInstanceUID = s.SOPInstanceUID
                ref.ReferencedInstanceSequence.append(item)
            return [ref]

        def create_shared_functional_groups(ref_dicom):
            shared = Dataset()
            measures = Dataset()
            measures.PixelSpacing = ref_dicom.PixelSpacing
            measures.SliceThickness = ref_dicom.SliceThickness
            shared.PixelMeasuresSequence = [measures]
            orientation = Dataset()
            orientation.ImageOrientationPatient = ref_dicom.ImageOrientationPatient
            shared.PlaneOrientationSequence = [orientation]
            return [shared]

        def create_per_frame_functional_groups(dicom_series, segments):
            per_frame = []
            for seg in segments:
                for s in dicom_series:
                    frame = Dataset()
                    frame.SegmentIdentificationSequence = [Dataset()]
                    frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber = seg.SegmentNumber
                    frame.PlanePositionSequence = [Dataset()]
                    frame.PlanePositionSequence[0].ImagePositionPatient = s.ImagePositionPatient
                    frame.DerivationImageSequence = [Dataset()]
                    frame.DerivationImageSequence[0].SourceImageSequence = [Dataset()]
                    frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPClassUID = s.SOPClassUID
                    frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID = s.SOPInstanceUID
                    per_frame.append(frame)
            return per_frame

        ds.ReferencedSeriesSequence = create_ref_series_sequence(dicom_series)
        ds.SharedFunctionalGroupsSequence = create_shared_functional_groups(ref_dicom)
        ds.PerFrameFunctionalGroupsSequence = create_per_frame_functional_groups(dicom_series, ds.SegmentSequence)
        ds.NumberOfFrames = len(ds.PerFrameFunctionalGroupsSequence)

        # --- Prepare Pixel Data ---
        pixel_data = []
        for segment_item in ds.SegmentSequence:
            label_id = next(k for k, v in self.label_mapping.items() if v == segment_item.SegmentLabel)
            binary_mask = (seg_data == label_id).astype(np.uint8)
            for i in range(seg_data.shape[2]):
                # NOTE: This transpose might be needed depending on the NIfTI orientation.
                # It converts NumPy's column-major slice to DICOM's row-major.
                frame_data = np.transpose(binary_mask[:, :, i])
                pixel_data.append(frame_data.tobytes())

        ds.PixelData = b''.join(pixel_data)

        ds.save_as(output_path, write_like_original=False)
        logger.info(f"✅ DICOM SEG file created successfully at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="A generalized converter for NIfTI segmentation masks to DICOM SEG format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--nifti-mask", required=True, help="Path to the input NIfTI segmentation mask file.")
    parser.add_argument("--dicom-dir", required=True, help="Path to the directory of the original DICOM series.")
    parser.add_argument("--output-file", required=True, help="Path for the output DICOM SEG file.")
    parser.add_argument("--config-json", required=True,
                        help="Path to the JSON configuration file for segmentation metadata.")

    args = parser.parse_args()

    if not os.path.exists(args.nifti_mask):
        raise FileNotFoundError(f"NIfTI mask file not found: {args.nifti_mask}")
    if not os.path.isdir(args.dicom_dir):
        raise NotADirectoryError(f"DICOM directory not found: {args.dicom_dir}")

    try:
        converter = NiftiToDicomSegConverter(args.config_json)
        converter.convert_to_dicom_seg(args.nifti_mask, args.dicom_dir, args.output_file)
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}", exc_info=True)


if __name__ == '__main__':
    main()
    # ```
#
# ---
#
## 2. Example JSON Configuration Files
#
# You
# can
# create
# a
# `.json
# ` file
# for each type of segmentation you want to perform.
#
### `heart_config.json` (For ACDC Cardiac Segmentation)
#
# ```json
# {
#     "seriesDescription": "Cardiac Segmentation (ACDC)",
#     "contentCreatorName": "HeartSeg AI",
#     "algorithmName": "nnUNet_Cardiac_ACDC",
#     "bodyPartExamined": "HEART",
#     "segments": {
#         "1": {
#             "description": "Right Ventricle",
#             "color": [0, 0, 255],
#             "sctCodes": {
#                 "category": {"CodeValue": "85756007", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Heart"},
#                 "type": {"CodeValue": "27532002", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Right ventricle"},
#                 "anatomicRegion": {"CodeValue": "80891009", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Heart"}
#             }
#         },
#         "2": {
#             "description": "Myocardium",
#             "color": [0, 255, 0],
#             "sctCodes": {
#                 "category": {"CodeValue": "85756007", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Heart"},
#                 "type": {"CodeValue": "78721008", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Myocardium"},
#                 "anatomicRegion": {"CodeValue": "80891009", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Heart"}
#             }
#         },
#         "3": {
#             "description": "Left Ventricle",
#             "color": [255, 0, 0],
#             "sctCodes": {
#                 "category": {"CodeValue": "85756007", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Heart"},
#                 "type": {"CodeValue": "72481000", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Left ventricle"},
#                 "anatomicRegion": {"CodeValue": "80891009", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Heart"}
#             }
#         }
#     }
# }
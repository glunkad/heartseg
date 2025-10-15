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
    Convert a single NIfTI segmentation mask into a DICOM SEG object.
    This version expects JSON metadata exported from dcmqi (qiicr.org/dcmqi) or a
    similar structure. It is tolerant to a few common JSON layouts.
    """

    def __init__(self, config_path):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load/parse configuration JSON: {e}")

        # Support a couple of JSON layouts:
        # Preferred: {"segments": { "1": {...}, "2": {...} }, ...}
        # dcmqi style can be under "SegmentDescription" or "segmentAttributes" etc.
        segments = {}
        if "segments" in self.config and isinstance(self.config["segments"], dict):
            segments = self.config["segments"]
        else:
            # try common alternative keys used by dcmqi exports
            for alt in ("segmentAttributes", "SegmentDescription", "segmentDescriptions", "segmentsDescriptions",
                        "SegmentSequence"):
                if alt in self.config and isinstance(self.config[alt], dict):
                    segments = self.config[alt]
                    break
                if alt in self.config and isinstance(self.config[alt], list):
                    # convert list to dict with numeric keys if possible
                    segs = self.config[alt]
                    tmp = {}
                    for i, s in enumerate(segs, start=1):
                        tmp[str(i)] = s
                    segments = tmp
                    break

        if not segments:
            # last resort: look for top-level "segment" list
            if "segment" in self.config and isinstance(self.config["segment"], list):
                tmp = {}
                for i, s in enumerate(self.config["segment"], start=1):
                    tmp[str(i)] = s
                segments = tmp

        if not segments:
            raise ValueError("No segments found in JSON config. Ensure it has a 'segments' or dcmqi-like structure.")

        # normalize segments
        normalized = {}
        for k, v in segments.items():
            try:
                label_id = int(k)
            except Exception:
                # try to read 'id' inside object
                label_id = int(v.get("id", k))
            normalized[str(label_id)] = v
        self.segments = normalized

        # mappings
        self.label_mapping = {}
        self.sct_codes = {}
        self.color_map = {}

        for k, v in self.segments.items():
            lid = int(k)
            # common fields: "name" or "description" or "label"
            name = v.get("description") or v.get("name") or v.get("label") or v.get("SegmentLabel")
            if not name:
                name = f"Segment{lid}"
            self.label_mapping[lid] = name

            # colors: either "color" or "recommendedDisplayCIELab" or "rgb"
            color_rgb = v.get("color") or v.get("recommendedDisplayRGB") or v.get("RGB")
            if isinstance(color_rgb, str):
                # try comma separated
                try:
                    color_rgb = [int(x) for x in color_rgb.split(",")]
                except Exception:
                    color_rgb = [128, 128, 128]
            if not color_rgb:
                color_rgb = [128, 128, 128]
            self.color_map[name] = color_rgb

            # codes: dcmqi may nest code attributes under different keys
            sct = {}
            candidate_code_keys = ("sctCodes", "codes", "ontology", "anatomicRegion", "type", "category",
                                   "SegmentedPropertyCategoryCodeSequence")
            if "sctCodes" in v:
                sct = v["sctCodes"]
            else:
                # try to compose from available fields
                # dcmqi often includes "typeCode", "anatomicRegionCode", "categoryCode"
                cat = v.get("categoryCode") or v.get("CategoryCode") or v.get("segmentedPropertyCategoryCode")
                typ = v.get("typeCode") or v.get("TypeCode") or v.get("segmentedPropertyTypeCode")
                anat = v.get("anatomicRegionCode") or v.get("AnatomicRegionCode") or v.get("anatomicRegion")
                composed = {}
                if cat:
                    composed["category"] = cat
                if typ:
                    composed["type"] = typ
                if anat:
                    composed["anatomicRegion"] = anat
                if composed:
                    sct = composed

            self.sct_codes[lid] = sct

        logger.info(f"Configuration loaded. Segments: {list(self.label_mapping.values())}")

    @staticmethod
    def dict_to_code_ds(code_dict):
        """
        Convert a simple code dict (CodeValue, CodingSchemeDesignator, CodeMeaning)
        into a pydicom Dataset suitable for inserting into a Code Sequence.
        """
        ds = Dataset()
        # accept a few naming variants
        cv = code_dict.get("CodeValue") or code_dict.get("codeValue") or code_dict.get("code") or code_dict.get("value")
        csd = code_dict.get("CodingSchemeDesignator") or code_dict.get("codingScheme") or code_dict.get("CodingScheme")
        cm = code_dict.get("CodeMeaning") or code_dict.get("codeMeaning") or code_dict.get("meaning") or code_dict.get(
            "CodeDescription")
        if cv is not None:
            ds.CodeValue = str(cv)
        if csd is not None:
            ds.CodingSchemeDesignator = str(csd)
        if cm is not None:
            ds.CodeMeaning = str(cm)
        return ds

    def load_dicom_series(self, dicom_dir):
        """Load and sort DICOM series robustly from a directory."""
        dicom_files = []
        for root, _, files in os.walk(dicom_dir):
            for filename in sorted(files):
                if filename.lower().endswith(('.dcm', '.ima', '.dic')) or '.' not in filename:
                    filepath = os.path.join(root, filename)
                    try:
                        ds = pydicom.dcmread(filepath, force=True, stop_before_pixels=True)
                        dicom_files.append((filepath, ds))
                    except Exception:
                        continue

        if not dicom_files:
            logger.info("No DICOM files read with stop_before_pixels=True, attempting full reads (may be slower).")
            for root, _, files in os.walk(dicom_dir):
                for filename in sorted(files):
                    if filename.lower().endswith(('.dcm', '.ima', '.dic')) or '.' not in filename:
                        filepath = os.path.join(root, filename)
                        try:
                            ds = pydicom.dcmread(filepath, force=True)
                            dicom_files.append((filepath, ds))
                        except Exception:
                            continue

        # sort by ImagePositionPatient (z) if available else InstanceNumber
        def sort_key(item):
            ds = item[1]
            if hasattr(ds, "ImagePositionPatient"):
                try:
                    return float(ds.ImagePositionPatient[2])
                except Exception:
                    pass
            if hasattr(ds, "InstanceNumber"):
                try:
                    return int(ds.InstanceNumber)
                except Exception:
                    pass
            return 0

        dicom_files.sort(key=sort_key)
        ds_list = [ds for fp, ds in dicom_files]
        logger.info(f"Loaded {len(ds_list)} DICOM files from '{dicom_dir}'")
        return ds_list

    def load_nifti_segmentation(self, nifti_path):
        nifti_img = nib.load(nifti_path)
        seg_data = nifti_img.get_fdata().astype(np.int32)
        logger.info(f"Loaded NIfTI segmentation '{os.path.basename(nifti_path)}' with shape: {seg_data.shape}")
        return seg_data

    def create_segment_sequence(self, unique_labels):
        segment_sequence = []

        def rgb_to_cielab_dicom(rgb_8bit):
            rgb_float = np.array(rgb_8bit, dtype=np.uint8).reshape(1, 1, 3) / 255.0
            lab_float = color.rgb2lab(rgb_float)
            L, a, b = lab_float[0][0]
            # scale to 0-65535 as unsigned 16 values used in some SEG attributes
            return [int((L / 100.0) * 65535), int(((a + 128.0) / 255.0) * 65535), int(((b + 128.0) / 255.0) * 65535)]

        for label in sorted(unique_labels):
            if label == 0:
                continue
            label_int = int(label)
            name = self.label_mapping.get(label_int, f"Seg{label_int}")
            codes = self.sct_codes.get(label_int, {})

            seg_ds = Dataset()
            seg_ds.SegmentNumber = len(segment_sequence) + 1
            seg_ds.SegmentLabel = name
            seg_ds.SegmentAlgorithmType = "AUTOMATIC"
            seg_ds.SegmentAlgorithmName = self.config.get("algorithmName", "AI Segmentation")

            assigned_color = self.color_map.get(name, [128, 128, 128])
            seg_ds.RecommendedDisplayCIELabValue = rgb_to_cielab_dicom(assigned_color)

            # add code sequences if available
            if "category" in codes and isinstance(codes["category"], dict):
                seg_ds.SegmentedPropertyCategoryCodeSequence = [self.dict_to_code_ds(codes["category"])]
            if "type" in codes and isinstance(codes["type"], dict):
                seg_ds.SegmentedPropertyTypeCodeSequence = [self.dict_to_code_ds(codes["type"])]
            if "anatomicRegion" in codes and isinstance(codes["anatomicRegion"], dict):
                seg_ds.AnatomicRegionSequence = [self.dict_to_code_ds(codes["anatomicRegion"])]

            segment_sequence.append(seg_ds)

        return segment_sequence

    def convert_to_dicom_seg(self, nifti_path, dicom_dir, output_path):
        dicom_series = self.load_dicom_series(dicom_dir)
        if not dicom_series:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

        seg_data = self.load_nifti_segmentation(nifti_path)

        # Expect seg_data shape to be either (X,Y,Z) or (X,Y,Z,1) or (X,Y,Z,Nlabels)
        if seg_data.ndim == 4 and seg_data.shape[3] == 1:
            seg_data = seg_data[..., 0]
        if seg_data.ndim != 3:
            raise ValueError("Unsupported NIfTI segmentation shape. Expected 3D array (X,Y,Z).")

        ref_dicom = dicom_series[0]

        unique_labels = [int(l) for l in np.unique(seg_data) if int(l) != 0]
        logger.info(f"Found unique labels in NIfTI mask: {unique_labels}")

        # Build generic file meta and dataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)

        # Basic patient/study copying
        for attr in ("PatientName", "PatientID", "StudyInstanceUID", "StudyDate", "StudyTime", "AccessionNumber"):
            if hasattr(ref_dicom, attr):
                setattr(ds, attr, getattr(ref_dicom, attr))

        ds.SeriesInstanceUID = generate_uid()
        try:
            ds.SeriesNumber = int(getattr(ref_dicom, "SeriesNumber", 0)) + 100
        except Exception:
            ds.SeriesNumber = 100
        ds.SeriesDescription = self.config.get("seriesDescription", "Segmentation")
        ds.Modality = "SEG"
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = 1
        ds.BodyPartExamined = self.config.get("bodyPartExamined", "")

        now = datetime.now()
        ds.InstanceCreationDate = now.strftime('%Y%m%d')
        ds.InstanceCreationTime = now.strftime('%H%M%S.%f')[:-3]
        ds.ContentDate = now.strftime('%Y%m%d')
        ds.ContentTime = now.strftime('%H%M%S.%f')[:-3]

        ds.ImageType = ["DERIVED", "SECONDARY"]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = int(getattr(ref_dicom, "Rows", seg_data.shape[0]))
        ds.Columns = int(getattr(ref_dicom, "Columns", seg_data.shape[1]))
        ds.BitsAllocated = 8
        ds.BitsStored = 1
        ds.HighBit = 0
        ds.PixelRepresentation = 0

        ds.SegmentationType = "BINARY"
        ds.ContentLabel = "SEGMENTATION"
        ds.ContentCreatorName = self.config.get("contentCreatorName", "AI Model")

        ds.SegmentSequence = self.create_segment_sequence(unique_labels)
        if not ds.SegmentSequence:
            raise ValueError("No valid segments were created. Check that labels in NIfTI match the config JSON.")

        # Helper functions to build sequences
        def create_ref_series_sequence(dicom_series):
            ref = Dataset()
            ref.SeriesInstanceUID = dicom_series[0].SeriesInstanceUID
            ref.ReferencedInstanceSequence = []
            for s in dicom_series:
                item = Dataset()
                item.ReferencedSOPClassUID = getattr(s, "SOPClassUID", "")
                item.ReferencedSOPInstanceUID = getattr(s, "SOPInstanceUID", "")
                ref.ReferencedInstanceSequence.append(item)
            return [ref]

        def create_shared_functional_groups(ref_dicom):
            shared = Dataset()
            measures = Dataset()
            if hasattr(ref_dicom, "PixelSpacing"):
                measures.PixelSpacing = ref_dicom.PixelSpacing
            if hasattr(ref_dicom, "SliceThickness"):
                measures.SliceThickness = ref_dicom.SliceThickness
            shared.PixelMeasuresSequence = [measures]

            orientation = Dataset()
            if hasattr(ref_dicom, "ImageOrientationPatient"):
                orientation.ImageOrientationPatient = ref_dicom.ImageOrientationPatient
            shared.PlaneOrientationSequence = [orientation]
            return [shared]

        def create_per_frame_functional_groups(dicom_series, segments):
            per_frame = []
            # We will create frames in order: for each segment -> for each slice (matching dicom_series)
            for seg in segments:
                for s in dicom_series:
                    frame = Dataset()
                    # Segment identification
                    sid = Dataset()
                    sid.ReferencedSegmentNumber = seg.SegmentNumber
                    frame.SegmentIdentificationSequence = [sid]
                    # Plane position: use ImagePositionPatient for the referenced slice
                    ppos = Dataset()
                    if hasattr(s, "ImagePositionPatient"):
                        ppos.ImagePositionPatient = s.ImagePositionPatient
                    if hasattr(s, "InstanceNumber"):
                        ppos.InstanceNumber = s.InstanceNumber
                    frame.PlanePositionSequence = [ppos]
                    # Derivation / source image reference
                    der = Dataset()
                    src = Dataset()
                    src.ReferencedSOPClassUID = getattr(s, "SOPClassUID", "")
                    src.ReferencedSOPInstanceUID = getattr(s, "SOPInstanceUID", "")
                    der.SourceImageSequence = [Dataset()]
                    der.SourceImageSequence[0].ReferencedSOPClassUID = src.ReferencedSOPClassUID
                    der.SourceImageSequence[0].ReferencedSOPInstanceUID = src.ReferencedSOPInstanceUID
                    frame.DerivationImageSequence = [der]
                    per_frame.append(frame)
            return per_frame

        ds.ReferencedSeriesSequence = create_ref_series_sequence(dicom_series)
        ds.SharedFunctionalGroupsSequence = create_shared_functional_groups(ref_dicom)
        ds.PerFrameFunctionalGroupsSequence = create_per_frame_functional_groups(dicom_series, ds.SegmentSequence)
        ds.NumberOfFrames = len(ds.PerFrameFunctionalGroupsSequence)

        # --- Prepare Pixel Data ---
        # Build binary frames. We expect seg_data orientation to match dicom_series ordering.
        # Commonly, nifti shape is (X=cols, Y=rows, Z=slices) or (rows, cols, slices) depending on source.
        # We'll attempt to match ref_dicom Rows/Columns to nifti dims by checking shapes and transposing if needed.
        nx, ny, nz = seg_data.shape
        rows = int(ds.Rows)
        cols = int(ds.Columns)

        # Determine if transpose is required: if nifti first two dims equal (rows, cols) or (cols, rows)
        transpose_needed = False
        if (nx == rows and ny == cols):
            transpose_needed = False
        elif (nx == cols and ny == rows):
            transpose_needed = True
        else:
            # fallback: try transpose to make it match
            transpose_needed = True

        pixel_bytes = []
        for seg_ds in ds.SegmentSequence:
            # find label id by reverse lookup
            label_id = None
            for k, v in self.label_mapping.items():
                if v == seg_ds.SegmentLabel:
                    label_id = int(k)
                    break
            if label_id is None:
                raise ValueError(f"Label id for segment '{seg_ds.SegmentLabel}' not found in mapping.")

            # build frames per slice for this label
            for slice_idx in range(nz):
                mask_slice = seg_data[:, :, slice_idx] == label_id
                if transpose_needed:
                    frame = np.transpose(mask_slice)
                else:
                    frame = mask_slice
                # Ensure frame matches DICOM Rows x Columns
                try:
                    frame = frame.astype(np.uint8)
                    if frame.shape != (rows, cols):
                        # try to reshape / pad / crop minimally
                        frows, fcols = frame.shape
                        out = np.zeros((rows, cols), dtype=np.uint8)
                        minr = min(rows, frows)
                        minc = min(cols, fcols)
                        out[:minr, :minc] = frame[:minr, :minc]
                        frame = out
                except Exception:
                    frame = np.zeros((rows, cols), dtype=np.uint8)

                # For simplicity, we will store each frame as one byte per pixel (0/1) and not pack bits.
                pixel_bytes.append(frame.tobytes())

        ds.PixelData = b"".join(pixel_bytes)

        # Save file (overwrite if exists)
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ds.save_as(output_path, write_like_original=False)
        logger.info(f"DICOM SEG file created successfully at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a single NIfTI segmentation mask to a DICOM SEG file using JSON metadata (dcmqi-style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--nifti-mask", required=True, help="Path to the input single NIfTI segmentation mask file.")
    parser.add_argument("--dicom-dir", required=True,
                        help="Path to the directory of the original DICOM series (reference).")
    parser.add_argument("--output-file", required=True, help="Path for the output DICOM SEG file (e.g. out/seg.dcm).")
    parser.add_argument("--config-json", required=True, help="Path to the JSON configuration file (from dcmqi).")

    args = parser.parse_args()

    if not os.path.exists(args.nifti_mask):
        raise FileNotFoundError(f"NIfTI mask file not found: {args.nifti_mask}")
    if not os.path.isdir(args.dicom_dir):
        raise NotADirectoryError(f"DICOM directory not found: {args.dicom_dir}")
    if not os.path.exists(args.config_json):
        raise FileNotFoundError(f"Config JSON not found: {args.config_json}")

    try:
        converter = NiftiToDicomSegConverter(args.config_json)
        converter.convert_to_dicom_seg(args.nifti_mask, args.dicom_dir, args.output_file)
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

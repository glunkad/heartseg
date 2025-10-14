import os, argparse, warnings, textwrap, torch, psutil, shutil
from fnmatch import fnmatch
import multiprocessing as mp
import nibabel as nib
from pathlib import Path
import importlib.resources
from tqdm import tqdm
from heartseg import *
from heartseg.init_inference import init_inference


warnings.filterwarnings("ignore")


def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
            This script runs inference using the trained HeartSeg nnUNet model.
            If not already installed, the script will download the pretrained models from the GitHub releases.
            If the input is a directory of DICOM files, it will be automatically preprocessed.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            heartseg input.nii.gz output_folder
            heartseg input_dicom_folder output_folder
            heartseg input_nifti_folder output_folder
            heartseg input_folder output_folder --suffix _0000
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input', type=Path,
        help='The input folder containing images to run the model on, or a single .nii.gz (or .nii) image.'
    )
    parser.add_argument(
        'output', type=Path,
        help='The output folder where the model outputs will be stored.'
    )
    parser.add_argument(
        '--iso', action="store_true", default=False,
        help='Use isotropic output as output by the model instead of resampling output to the input, defaults to false.'
    )
    parser.add_argument(
        '--suffix', '-s', type=str, nargs='+', default=[''],
        help='Suffix to use for the input images, defaults to "".'
    )
    parser.add_argument(
        '--data-dir', '-d', type=Path, default=None,
        help='The path to store the nnUNet data.'
    )
    parser.add_argument(
        '--no-stalling', action="store_true", default=False,
        help='Set multiprocessing method to "forkserver" to avoid deadlock issues, default to False.'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=os.cpu_count(),
        help=f'Max worker to run in parallel proccess, defaults to numer of available cores'
    )
    parser.add_argument(
        '--max-workers-nnunet', type=int,
        default=int(max(min(os.cpu_count(), psutil.virtual_memory().total / 2 ** 30 // 8), 1)),
        help='Max worker to run in parallel proccess for nnUNet, defaults to min(numer of available cores, Memory in GB / 8).'
    )
    parser.add_argument(
        '--device', type=str, choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run the nnUNet model on, defaults to "cuda" if available, otherwise "cpu".'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    input_path = args.input
    output_path = args.output
    output_iso = args.iso
    suffix = args.suffix
    max_workers = args.max_workers
    max_workers_nnunet = min(args.max_workers_nnunet, max_workers)
    device = args.device
    quiet = args.quiet

    # Init data_path
    if args.data_dir is not None:
        data_path = args.data_dir
    elif 'HEARTSEG_DATA' in os.environ:
        data_path = Path(os.environ.get('HEARTSEG_DATA', ''))
    else:
        data_path = importlib.resources.files(models)

    # Change multiprocessing method if specified
    if args.no_stalling:
        mp.set_start_method('forkserver', force=True)

    # Default release to use
    default_release = list(ZIP_URLS.values())[0].split('/')[-2]

    # Install weights if not present
    init_inference(
        data_path=data_path,
        dict_urls=ZIP_URLS,
        quiet=quiet
    )

    # Run inference
    inference(
        input_path=input_path,
        output_path=output_path,
        data_path=data_path,
        default_release=default_release,
        output_iso=output_iso,
        suffix=suffix,
        max_workers=max_workers,
        max_workers_nnunet=max_workers_nnunet,
        device=device,
        quiet=quiet
    )


def inference(
        input_path,
        output_path,
        data_path,
        default_release,
        output_iso=False,
        suffix=[''],
        max_workers=os.cpu_count(),
        max_workers_nnunet=int(max(min(os.cpu_count(), psutil.virtual_memory().total / 2 ** 30 // 8), 1)),
        device='cuda',
        quiet=False
):
    '''
    Inference function for heart segmentation.

    Parameters
    ----------
    input_path : pathlib.Path or string
        The input folder path containing the niftii images.
    output_path : pathlib.Path or string
        The output folder path that will contain the predictions.
    data_path : pathlib.Path or string
        Folder path containing the network weights.
    default_release : string
        Default release used for inference.
    output_iso : bool
        If False, output predictions will be resampled to the original space.
    suffix : list of string
        Suffix to use for the input images.
    max_workers : int
        Max worker to run in parallel proccess, defaults to numer of available cores.
    max_workers_nnunet : int
        Max worker to run in parallel proccess for nnUNet.
    device : 'cuda' or 'cpu'
        Device to run the nnUNet model on.
    quiet : bool
        If True, will reduce the amount of displayed information.

    Returns
    -------
    list of string
        List of output folders.
    '''
    # Convert paths to Path like objects
    if isinstance(input_path, str):
        input_path = Path(input_path)
    elif not isinstance(input_path, Path):
        raise ValueError('input_path should be a Path object from pathlib or a string')

    if isinstance(output_path, str):
        output_path = Path(output_path)
    elif not isinstance(output_path, Path):
        raise ValueError('output_path should be a Path object from pathlib or a string')

    if isinstance(data_path, str):
        data_path = Path(data_path)
    elif not isinstance(data_path, Path):
        raise ValueError('data_path should be a Path object from pathlib or a string')

    # Check if the data folder exists
    if not data_path.exists():
        raise FileNotFoundError(f"The heartseg data folder does not exist at {data_path}.")

    # Preprocess if input is a directory of DICOMs
    if input_path.is_dir():
        # Heuristic: if the directory contains no NIfTI files, assume it's DICOM and preprocess it.
        if not any(input_path.rglob('*.nii*')):
            if not quiet:
                print("\nInput directory contains no NIfTI files. Assuming DICOM input and starting preprocessing...")

            preprocessed_path = output_path / 'preprocessed_input'
            preprocessed_path.mkdir(exist_ok=True, parents=True)

            # Call the preprocessing function
            preprocess_data(input_path, preprocessed_path)

            input_path = preprocessed_path  # Update input_path to the new nifti dir
            if not quiet:
                print(f"Preprocessing complete. Using NIfTI files from '{preprocessed_path}' for inference.")

    # Datasets data
    my_dataset = 'Dataset101_HeartSeg'
    fold = 0

    # Set nnUNet results path
    nnUNet_results = data_path / 'nnUNet' / 'results'

    # Load device
    if isinstance(device, str):
        assert device in ['cpu', 'cuda',
                          'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
        if device == 'cpu':
            torch.set_num_threads(mp.cpu_count())
            device = torch.device('cpu')
        elif device == 'cuda':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device('cuda')
        else:
            device = torch.device('mps')
    else:
        assert isinstance(device, torch.device)

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running HeartSeg with the following parameters:
            input = "{input_path}"
            output = "{output_path}"
            iso = {output_iso}
            suffix = {suffix}
            data_dir = "{data_path}"
            max_workers = {max_workers}
            max_workers_nnunet = {max_workers_nnunet}
            device = "{device.type}"
        '''))

    if not quiet: print('\n' 'Making input dir with _0000 suffix:')
    if not input_path.is_dir():
        # If the input is a single file, copy it to the input_raw folder
        (output_path / 'input_raw').mkdir(parents=True, exist_ok=True)

        # Check suffixes
        if input_path.name.endswith(".nii.gz"):
            # Copy file
            dst_path = output_path / 'input_raw' / input_path.name.replace('.nii.gz', '_0000.nii.gz')
            shutil.copy(input_path, dst_path)
        elif input_path.suffix == ".nii":
            # Compress file
            src_img = nib.load(input_path)
            dst_path = output_path / 'input_raw' / input_path.name.replace('.nii', '_0000.nii.gz')
            nib.save(src_img, dst_path)
        else:
            raise ValueError(f"Unknown file type: {''.join(input_path.suffixes)}, please use niftii files")
    else:
        # If the input is a folder, copy the files to the input_raw folder
        cpdir_mp(
            input_path,
            output_path / 'input_raw',
            pattern=sum(
                [[f'*{s}.nii.gz', f'sub-*/anat/*{s}.nii.gz', f'*{s}.nii', f'sub-*/anat/*{s}.nii'] for s in suffix], []),
            flat=True,
            replace={'.nii.gz': '_0000.nii.gz'},
            compress=True,
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

    if not quiet: print('\n' 'Copying the input images to the input folder for processing:')
    cpdir_mp(
        output_path / 'input_raw',
        output_path / 'input',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    # Get the nnUNet parameters from the results folder
    nnUNetTrainer, nnUNetPlans, configuration = next(
        (nnUNet_results / my_dataset).glob('*/fold_*')).parent.name.split('__')
    # Check if the final checkpoint exists, if not use the latest checkpoint
    checkpoint = 'checkpoint_final.pth' if (
            nnUNet_results / my_dataset / f'{nnUNetTrainer}__{nnUNetPlans}__{configuration}' / f'fold_{fold}' / 'checkpoint_final.pth').is_file() else 'checkpoint_best.pth'

    # Construct model folder
    model_folder = nnUNet_results / my_dataset / f'{nnUNetTrainer}__{nnUNetPlans}__{configuration}'

    if not quiet: print('\n' 'Running model:')
    predict_nnunet(
        model_folder=model_folder,
        images_dir=output_path / 'input',
        output_dir=output_path / 'output',
        folds=str(fold),
        save_probabilities=True,
        checkpoint=checkpoint,
        npp=max_workers_nnunet,
        nps=max_workers_nnunet,
        device=device
    )

    # Remove unnecessary files from output folder
    (output_path / 'output' / 'dataset.json').unlink(missing_ok=True)
    (output_path / 'output' / 'plans.json').unlink(missing_ok=True)
    (output_path / 'output' / 'predict_from_raw_data_args.json').unlink(missing_ok=True)
    for f in (output_path / 'output').glob('*.pkl'):
        f.unlink(missing_ok=True)

    # Remove the input_raw folder
    shutil.rmtree(output_path / 'input_raw', ignore_errors=True)

    # Remove the input folder
    if not output_iso:
        shutil.rmtree(output_path / 'input', ignore_errors=True)

    # Return list of output paths
    return [str(output_path / folder) for folder in os.listdir(str(output_path))]


if __name__ == '__main__':
    main()

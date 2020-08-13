from typing import Tuple, Dict, Any, Union, List

from .unlabeled_dataset import UnlabeledDataLoader, UnlabeledDataset


def load_unpaired_unlabeled_datasets(
    dir_A: Union[str, List[str]],
    dir_B: Union[str, List[str]],
    max_dataset_size: int,
    batch_size: int,
    serial_batches: bool,
    num_threads: int,
    grayscale_A: bool,
    grayscale_B: bool,
    transform_properties: Dict[str, Any],
) -> Tuple[UnlabeledDataLoader, UnlabeledDataLoader]:
    """Create dataloader for two unpaired and unlabeled datasets.

    E.g. used by cycle gan with data from two domains.

    Args:
        dir_A: path to images of domain A
        dir_B: path to images of domain B
        max_dataset_size (int): maximum amount of images to load; -1 means
            infinity
        batch_size (int): input batch size
        serial_batches (bool): if true, takes images in order to make batches,
            otherwise takes them randomly
        num_threads (int): threads for loading data
        grayscale_A (bool): transform domain A to gray images
        grayscale_B (bool): transform domain B to gray images
        transform_properties: dict containing properties for transforming images
    """
    max_dataset_size = float("inf") if max_dataset_size == -1 else max_dataset_size

    transform_properties["grayscale"] = grayscale_A
    A = UnlabeledDataset(dir_A, max_dataset_size, transform_properties)
    transform_properties["grayscale"] = grayscale_B
    B = UnlabeledDataset(dir_B, max_dataset_size, transform_properties)

    # Transform datasets into dataloaders.
    A = UnlabeledDataLoader(
        dataset=A,
        max_dataset_size=max_dataset_size,
        batch_size=batch_size,
        serial_batches=serial_batches,
        num_threads=num_threads,
    )
    B = UnlabeledDataLoader(
        dataset=B,
        max_dataset_size=max_dataset_size,
        batch_size=batch_size,
        serial_batches=serial_batches,
        num_threads=num_threads,
    )

    return A, B

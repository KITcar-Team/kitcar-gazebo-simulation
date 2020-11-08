import os
from typing import Any, Dict, List, Tuple, Union

from .data_loader import DataLoader
from .labeled_dataset import LabeledDataset
from .unlabeled_dataset import UnlabeledDataset


def load_unpaired_unlabeled_datasets(
    dir_a: Union[str, List[str]],
    dir_b: Union[str, List[str]],
    max_dataset_size: int,
    batch_size: int,
    sequential: bool,
    num_threads: int,
    grayscale_a: bool,
    grayscale_b: bool,
    transform_properties: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloader for two unpaired and unlabeled datasets.

    E.g. used by cycle gan with data from two domains.

    Args:
        dir_a: path to images of domain a
        dir_b: path to images of domain b
        max_dataset_size (int): maximum amount of images to load; -1 means infinity
        batch_size (int): input batch size
        sequential (bool): if true, takes images in order, otherwise takes them randomly
        num_threads (int): threads for loading data
        grayscale_a (bool): transform domain a to gray images
        grayscale_b (bool): transform domain b to gray images
        transform_properties: dict containing properties for transforming images
    """
    max_dataset_size = float("inf") if max_dataset_size == -1 else max_dataset_size

    transform_properties["grayscale"] = grayscale_a
    a = UnlabeledDataset(transform_properties, dir_a)
    transform_properties["grayscale"] = grayscale_b
    b = UnlabeledDataset(transform_properties, dir_b)

    # Transform datasets into dataloaders.
    a = DataLoader(
        dataset=a,
        max_dataset_size=max_dataset_size,
        batch_size=batch_size,
        num_threads=num_threads,
        sequential=sequential,
    )
    b = DataLoader(
        dataset=b,
        max_dataset_size=max_dataset_size,
        batch_size=batch_size,
        num_threads=num_threads,
        sequential=sequential,
    )

    return a, b


def sample_generator(
    dataloader: DataLoader,
    n_samples: int = float("inf"),
):
    """Generator that samples from a dataloader.

    Args:
        dataloader: Dataloader.
        n_samples: Number of batches of samples.
    """
    iter_ = iter(dataloader)
    i = 0
    while i < n_samples:
        i += 1
        try:
            next_ = next(iter_)
        except StopIteration:
            iter_ = iter(dataloader)
            next_ = next(iter_)
        yield next_


def unpaired_sample_generator(
    dataloader_a: DataLoader,
    dataloader_b: DataLoader,
    n_samples: int = float("inf"),
):
    """Generator that samples pairwise from both dataloaders.

    Args:
        dataloader_a: Domain a dataloader.
        dataloader_b: Domain b dataloader.
        n_samples: Number of batches of samples.
    """
    iter_a = iter(dataloader_a)
    iter_b = iter(dataloader_b)
    i = 0
    while i < n_samples:
        i += 1
        try:
            next_a = next(iter_a)
        except StopIteration:
            iter_a = iter(dataloader_a)
            next_a = next(iter_a)
        try:
            next_b = next(iter_b)
        except StopIteration:
            iter_b = iter(dataloader_b)
            next_b = next(iter_b)
        yield next_a, next_b


def load_labeled_dataset(
    label_file: str,
    max_dataset_size: int,
    batch_size: int,
    sequential: bool,
    num_threads: int,
    transform_properties: Dict[str, Any],
) -> DataLoader:
    """Create dataloader for a labeled dataset.

    Args:
        label_file: Path to a file containing all labels
        max_dataset_size: Maximum amount of images to load; -1 means infinity
        batch_size: Batch size
        sequential: If true, takes images in order, otherwise takes them randomly
        num_threads: Threads for loading data
    """
    max_dataset_size = float("inf") if max_dataset_size == -1 else max_dataset_size

    dataset = LabeledDataset.from_yaml(label_file)
    dataset.transform_properties = transform_properties
    dataset._base_path = os.path.dirname(label_file)

    # Transform datasets into dataloaders.
    return DataLoader(
        dataset=dataset,
        max_dataset_size=max_dataset_size,
        batch_size=batch_size,
        num_threads=num_threads,
        sequential=sequential,
    )

from typing import Tuple, Dict, Any, Union, List

from .unlabeled_dataset import UnlabeledDataLoader, UnlabeledDataset


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
) -> Tuple[UnlabeledDataLoader, UnlabeledDataLoader]:
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
    a = UnlabeledDataset(dir_a, max_dataset_size, transform_properties)
    transform_properties["grayscale"] = grayscale_b
    b = UnlabeledDataset(dir_b, max_dataset_size, transform_properties)

    # Transform datasets into dataloaders.
    a = UnlabeledDataLoader(
        dataset=a,
        max_dataset_size=max_dataset_size,
        batch_size=batch_size,
        num_threads=num_threads,
        sequential=sequential,
    )
    b = UnlabeledDataLoader(
        dataset=b,
        max_dataset_size=max_dataset_size,
        batch_size=batch_size,
        num_threads=num_threads,
        sequential=sequential,
    )

    return a, b


def sample_generator(
    dataloader_a: UnlabeledDataLoader,
    dataloader_b: UnlabeledDataLoader,
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

from typing import Union, List, Tuple, Dict, Any

import torch
import torchvision.transforms as transforms
from PIL import Image

from .base_dataset import (
    BaseDataset,
    get_transform,
)
from .image_folder import make_dataset


class UnlabeledDataset(BaseDataset):
    """This dataset class can load a set of unlabeled data."""

    folder_path: Union[str, List[str]]
    """Maximum number of data points in the dataset."""
    transform_properties: Dict[str, Any]
    """Properties passed as arguments to transform generation function."""

    def __init__(self, folder_path, transform_properties):
        self.folder_path = folder_path
        self.transform_properties = transform_properties
        self.file_paths = self.load_file_paths()

    def load_file_paths(self) -> List[str]:
        """List[str]: File paths to all data."""
        if isinstance(self.folder_path, list):
            data = sum((make_dataset(d) for d in self.folder_path), [])
        else:
            data = make_dataset(self.folder_path)
        return data

    @property
    def transform(self) -> transforms.Compose:
        """transforms.Compose: Transformation that can be applied to images."""
        return get_transform(**self.transform_properties)

    def __getitem__(self, index) -> Tuple[Image.Image, str]:
        """Return a data point and its metadata information.

        Args:
            index: Index of returned datapoint.

        Returns:
            Datapoint + path to its file
        """
        path = self.file_paths[
            index % self.__len__()
        ]  # make sure index is within then range
        img = Image.open(path).convert("RGB")
        # apply image transformation
        img = self.transform(img)

        return img, path

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.file_paths)


class UnlabeledDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(
        self,
        dataset: UnlabeledDataset,
        batch_size: int = 1,
        serial_batches: bool = False,
        num_threads: int = 1,
    ):
        """Initialize this class

        Step 1: create a dataset instance given the name
        Step 2: create a multi-threaded data loader.
        """
        self.dataset = dataset
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=num_threads,
        )
        self.batch_size = batch_size

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data

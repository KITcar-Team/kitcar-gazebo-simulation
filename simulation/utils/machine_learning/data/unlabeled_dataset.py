from dataclasses import dataclass, field
from typing import List, Tuple, Union

from PIL import Image

from .base_dataset import BaseDataset
from .image_folder import find_images


@dataclass
class UnlabeledDataset(BaseDataset):
    """This dataset class can load a set of unlabeled data."""

    folder_path: Union[str, List[str]] = field(default_factory=list)
    """Path[s] to folders that contain the data."""

    def __post_init__(self):
        self.file_paths = self.load_file_paths()

    def load_file_paths(self) -> List[str]:
        """List[str]: File paths to all data."""
        if isinstance(self.folder_path, list):
            data = sum((find_images(d) for d in self.folder_path), [])
        else:
            data = find_images(self.folder_path)
        return data

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

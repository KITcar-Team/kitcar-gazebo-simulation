import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from torch import Tensor

from simulation.utils.basics.init_options import InitOptions
from simulation.utils.basics.save_options import SaveOptions

from .base_dataset import BaseDataset
from .image_folder import find_images


@dataclass
class LabeledDataset(BaseDataset, InitOptions, SaveOptions):
    """Dataset of images with labels."""

    attributes: Sequence[str] = None
    """Description of what each label means.

    Similar to headers in a table.
    """

    classes: Dict[int, str] = field(default_factory=dict)
    """Description of what the class ids represent."""

    labels: Dict[str, List[Sequence[Any]]] = field(default_factory=dict)
    """Collection of all labels structured as a dictionary."""

    _base_path: str = None
    """Path to the root of the dataset.

    Only needs to be set if the dataset is used to load data.
    """

    @property
    def available_files(self) -> List[str]:
        return [os.path.basename(file) for file in find_images(self._base_path)]

    def __getitem__(self, index) -> Tuple[Union[np.ndarray, Tensor], List[Any]]:
        """Return an image and it's label.

        Args:
            index: Index of returned datapoint.
        """
        key = self.available_files[index]
        label = self.labels.get(key, [-1])
        path = os.path.join(self._base_path, key)
        img = Image.open(path).convert("RGB")

        # apply image transformation
        img = self.transform(img)

        return img, label

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.labels)

    def filter_labels(self):
        """Remove labels that have no corresponding image."""
        all_files = self.available_files
        self.labels = {key: label for key, label in self.labels.items() if key in all_files}

    def append_label(self, key: str, label: Any):
        """Add a new label to the dataset.

        A single image (or any abstract object) can have many labels.
        """
        if key not in self.labels:
            self.labels[key] = []
        self.labels[key].append(label)

    def save_as_yaml(self, file_path):
        # Override the default method to temporarily remove base_path and prevent
        # writing it to the yaml file.
        bp = self._base_path
        del self._base_path
        super().save_as_yaml(file_path)
        self._base_path = bp

    @classmethod
    def from_yaml(cls, file):
        instance = cls._from_yaml(cls, file)
        instance._base_path = os.path.dirname(file)
        return instance

import itertools
import math
import os
import random
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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

    attributes: Optional[Sequence[str]] = None
    """Description of what each label means.

    Similar to headers in a table.
    """

    classes: Dict[int, str] = field(default_factory=dict)
    """Description of what the class ids represent."""

    labels: Dict[str, List[Sequence[Any]]] = field(default_factory=dict)
    """Collection of all labels structured as a dictionary."""

    _base_path: Optional[str] = None
    """Path to the root of the dataset.

    Only needs to be set if the dataset is used to load data.
    """

    @property
    def available_files(self) -> List[str]:
        return [
            os.path.basename(file)
            for file in find_images(self._base_path)
            if os.path.exists(file) and "debug" not in file
        ]

    def __getitem__(self, index: int) -> Tuple[Union[np.ndarray, Tensor], List[Any]]:
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

    def append_label(self, key: str, label: List[Sequence[Any]]):
        """Add a new label to the dataset.

        A single image (or any abstract object) can have many labels.
        """
        if key not in self.labels:
            self.labels[key] = []
        self.labels[key].append(label)

    def save_as_yaml(self, file_path: str):
        """Save the dataset to a yaml file. Override the default method to temporarily
        remove base_path and prevent writing it to the yaml file.

        Args:
            file_path: The output file.
        """
        bp = self._base_path
        del self._base_path
        super().save_as_yaml(file_path)
        self._base_path = bp

    def make_ids_continuous(self):
        """Reformat dataset to have continuous class ids."""
        ids = sorted(self.classes.keys())
        for new_id, old_id in enumerate(ids):
            self.replace_id(old_id, new_id)

    def replace_id(self, search_id: int, replace_id: int):
        """Replace id (search) with another id (replace) in the whole dataset.

        Args:
            search_id: The id being searched for.
            replace_id: The replacement id that replaces the search ids
        """
        # Replace in classes dict
        self.classes[replace_id] = self.classes.pop(search_id)

        # Replace in labels dict
        index = self.attributes.index("class_id")
        for label in itertools.chain(*self.labels.values()):
            if label[index] == search_id:
                label[index] = replace_id

    def split(self, fractions: List[float], shuffle: bool = True) -> List["LabeledDataset"]:
        """Split this dataset into multiple."""
        assert sum(fractions) == 1.0, "Fractions should sum up to 1"

        new_datasets = [
            LabeledDataset(attributes=self.attributes, classes=self.classes)
            for _ in fractions
        ]

        labels = list(self.labels.items())
        if shuffle:
            random.shuffle(labels)

        counts = (int(math.ceil(len(self) * frac)) for frac in ([0] + fractions))
        indices = list(accumulate(counts))
        start_indices = indices[:-1]
        end_indices = indices[1:]

        # end_indices[-1] could be bigger than len(self)
        # But even if it is larger, Python would make sure
        # that all labels are used exactly once.
        # So we do not need a test for it.

        for dataset, from_index, to_index in zip(new_datasets, start_indices, end_indices):
            dataset.labels = dict(labels[from_index:to_index])

        return new_datasets

    @classmethod
    def from_yaml(cls, file: str) -> "LabeledDataset":
        """Load a Labeled Dataset from a yaml file.

        Args:
            file: The path to the yaml file to load
        """
        instance = cls._from_yaml(cls, file)
        instance._base_path = os.path.dirname(file)
        return instance

    @classmethod
    def split_file(
        cls, file: str, parts: Dict[str, float], shuffle: bool = True
    ) -> List["LabeledDataset"]:
        """Split a dataset file into multiple datasets.

        Args:
            file: The path to the yaml file which gets split
            parts: A dict of names and and fractions
            shuffle: Split the labels randomly
        """
        # Read dataset and split it
        dataset = LabeledDataset.from_yaml(file)
        new_datasets = dataset.split(list(parts.values()), shuffle)

        # Save the split yaml files
        for name, dataset in zip(parts.keys(), new_datasets):
            dataset.save_as_yaml(os.path.join(os.path.dirname(file), f"{name}.yaml"))

        return new_datasets

    @classmethod
    def filter_file(cls, file: str) -> "LabeledDataset":
        """Filter broken file dependencies of a yaml file.

        Args:
            file: The path to the yaml file to filter
        """
        labeled_dataset = LabeledDataset.from_yaml(file)
        labeled_dataset.filter_labels()
        labeled_dataset.save_as_yaml(file)
        return labeled_dataset

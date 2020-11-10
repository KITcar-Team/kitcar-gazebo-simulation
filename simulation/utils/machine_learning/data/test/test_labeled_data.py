import os

import torch

from .. import image_folder, load_labeled_dataset, sample_generator
from ..labeled_dataset import LabeledDataset

LABEL_FILE_PATH = os.path.dirname(__file__) + "/labeled_dir/labels.yaml"


image_folder.IMG_EXTENSIONS.append("_png")  # Ensure that weird test images are found


def test_creating_labels():
    label_dataset = LabeledDataset(_base_path=os.path.dirname(LABEL_FILE_PATH))
    label_dataset.classes = {0: "class_0", 1: "class_1"}
    label_dataset.attributes = ["img_path", "class_id"]

    label_dataset.append_label("img0._png", ["img0._png", 0])
    label_dataset.append_label("img0._png", ["img0._png", 1])
    label_dataset.append_label("img1._png", ["img1._png", 1])

    label_dataset.save_as_yaml(LABEL_FILE_PATH)

    loaded_label_dataset = LabeledDataset.from_yaml(LABEL_FILE_PATH)

    assert label_dataset == loaded_label_dataset


def test_labeled_dataset():
    dataset = LabeledDataset.from_yaml(LABEL_FILE_PATH)

    assert dataset.__getitem__(0)[1] == [["img0._png", 0], ["img0._png", 1]]
    assert dataset.__getitem__(1)[1] == [["img1._png", 1]]
    assert len(dataset) == 2


def test_loading_labeled_dataset():
    dataloader = load_labeled_dataset(
        LABEL_FILE_PATH, 1, 1, True, 1, transform_properties={}
    )
    assert len(dataloader) == 1

    dataloader = load_labeled_dataset(
        LABEL_FILE_PATH, -1, 1, True, 1, transform_properties={}
    )
    assert len(dataloader) == 2

    sample_gen = sample_generator(dataloader, 5)
    assert len([0 for _ in sample_gen]) == 5

    sample_gen = sample_generator(dataloader, 5)
    # Check if first result is correct
    img, labels = next(sample_gen)

    assert isinstance(img, torch.Tensor)
    # Somewhat weird behavior of the dataloader:
    assert labels == [
        [("img0._png",), torch.Tensor([0])],
        [("img0._png",), torch.Tensor([1])],
    ]


def main():
    test_creating_labels()
    test_labeled_dataset()
    test_loading_labeled_dataset()
    os.remove(LABEL_FILE_PATH)


if __name__ == "__main__":
    main()

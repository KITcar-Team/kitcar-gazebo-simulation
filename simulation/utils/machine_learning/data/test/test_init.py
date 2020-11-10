import os

from .. import DataLoader, image_folder, load_unpaired_unlabeled_datasets

image_folder.IMG_EXTENSIONS.append("_png")  # Ensure that weird test images are found


def test_load_unpaired_unlabeled_datasets():
    tf_properties = {
        "load_size": 1,
        "crop_size": 1,
        "preprocess": None,
        "mask": None,
        "no_flip": True,
    }

    loader_a, loader_b = load_unpaired_unlabeled_datasets(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "labeled_dir"),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "labeled_dir"),
        1,
        1,
        False,
        1,
        False,
        False,
        tf_properties,
    )

    assert isinstance(loader_a, DataLoader), "Wrong type!"
    assert isinstance(loader_b, DataLoader), "Wrong type!"
    assert len(loader_a) == 1, "Exceed max dataset size"
    assert len(loader_b) == 1, "Exceed max dataset size"


def main():
    test_load_unpaired_unlabeled_datasets()


if __name__ == "__main__":
    main()

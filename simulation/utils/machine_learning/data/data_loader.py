import torch


class DataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading."""

    def __init__(
        self,
        dataset,
        max_dataset_size: int,
        batch_size: int = 1,
        num_threads: int = 1,
        sequential: bool = False,
    ):
        """Initialize this class.

        Step 1: create a dataset instance given the name
        Step 2: create a multi-threaded data loader.

        Args:
            dataset (UnlabeledDataset): the dataset to load
            max_dataset_size (int): the maximum amount of images to load
            batch_size (int): the input batch size
            num_threads (int): threads for loading data
            sequential (bool): if true, takes images in correct order (file path names),
        """
        self.dataset = dataset
        print("dataset [%s] was created" % type(self.dataset).__name__)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not sequential,
            num_workers=num_threads,
        )

        self.batch_size = batch_size
        self.max_dataset_size = max_dataset_size

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset."""
        return min(len(self.dataset), self.max_dataset_size)

    def __iter__(self):
        """Return a batch of data."""
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_dataset_size:
                break
            yield data

import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from utils import paint, plot_pie, plot_segment

__all__ = ["SensorDataset"]


class SensorDataset(Dataset):
    """
    A dataset class for multi-channel time-series data captured by wearable sensors.
    This class is slightly modified from the original implementation at:
     https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate
    """

    def __init__(
        self,
        dataset,
        window,
        stride,
        stride_test,
        path_processed,
        prefix,
        verbose=False,
    ):
        """
        Initialize instance.
        :param dataset: str. Name of target dataset.
        :param window: int. Sliding window size in samples.
        :param stride: int. Step size of the sliding window for training and validation data.
        :param stride_test: int. Step size of the sliding window for testing data.
        :param path_processed: str. Path to directory containing processed training, validation and test data.
        :param prefix: str. Prefix for the filename of the processed data. Options 'train', 'val', or 'test'.
        :param verbose: bool. Whether to print detailed information about the dataset when initializing.
        """

        self.dataset = dataset
        self.window = window
        self.stride = stride
        self.prefix = prefix
        self.path_processed = path_processed
        self.verbose = verbose

        self.path_dataset = os.path.join(path_processed, f"{prefix}_data.npz")

        if prefix == "test" and stride_test == 1:
            self.path_dataset = os.path.join(path_processed, "test_sample_wise.npz")
        else:
            self.path_dataset = os.path.join(path_processed, f"{prefix}_data.npz".format(prefix))

        dataset = np.load(self.path_dataset)

        self.data = dataset["data"]
        self.target = dataset["target"]
        self.len = self.data.shape[0]
        assert self.data.shape[0] == self.target.shape[0]
        print(
            paint(
                f"Creating {self.dataset} {self.prefix} HAR dataset of size {self.len} ..."
            )
        )

        if self.verbose:
            self.get_info()
            self.get_distribution()

        if prefix == "train":
            self.weight_samples = self.get_weights()

        self.n_channels = self.data.shape[-1]
        self.n_classes = self.target.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        data = torch.FloatTensor(self.data[index])
        target = torch.LongTensor([int(self.target[index])])
        idx = torch.from_numpy(np.array(index))

        return data, target, idx

    def get_info(self, n_samples=3):
        print(paint(f"[-] Information on {self.prefix} dataset:"))
        print("\t data: ", self.data.shape, self.data.dtype, type(self.data))
        print("\t target: ", self.target.shape, self.target.dtype, type(self.target))

        target_idx = [np.where(self.target == label)[0] for label in set(self.target)]
        target_idx_samples = np.array(
            [np.random.choice(idx, n_samples, replace=False) for idx in target_idx]
        ).flatten()

        for i, random_idx in enumerate(target_idx_samples):
            data, target, index = self.__getitem__(random_idx)
            if i == 0:
                print(paint(f"[-] Information on segment #{random_idx}/{self.len}:"))
                print("\t data: ", data.shape, data.dtype, type(data))
                print("\t target: ", target.shape, target.dtype, type(target))
                print("\t index: ", index, index.shape, index.dtype, type(index))

            path_save = os.path.join(self.path_processed, "segments")
            plot_segment(
                data,
                target,
                index=index,
                prefix=self.prefix,
                path_save=path_save,
                num_class=len(target_idx),
            )

    def get_distribution(self):
        plot_pie(
            self.target, self.prefix, os.path.join(self.path_processed, "distribution")
        )

    def get_weights(self):

        target = self.target

        target_count = np.array([np.sum(target == label) for label in set(target)])
        weight_target = 1.0 / target_count
        weight_samples = np.array([weight_target[t] for t in target])
        weight_samples = torch.from_numpy(weight_samples)
        weight_samples = weight_samples.double()

        return weight_samples


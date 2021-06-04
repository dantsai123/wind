import os
from typing import List, Tuple

import torch
import xarray as xr
from torch import Tensor
from torch.utils.data import Dataset

DATA_ROOT = "/home/oscar/nobackup/datasets/UKCP18"
DURATION = 36000
WIDTH = 82
HEIGHT = 112
N_GUST_BUCKETS = 10
FIELD_NAMES = ["psl", "uas", "vas", "sfcWind", "pr"]


class UKCP18(Dataset):
    def __init__(self, training: bool) -> None:
        self._n = DURATION
        self._d = len(FIELD_NAMES)

        print("Loading dataset...")
        input_data = [load_field(name)[1].values for name in FIELD_NAMES]
        print("Loaded")
        input_tensors = [torch.tensor(d) for d in input_data]
        inputs = torch.stack(input_tensors, axis=4)
        inputs = inputs.reshape(DURATION, HEIGHT, WIDTH, self._d).permute(0, 3, 1, 2)
        target_reals = torch.tensor(load_field("wsgsmax")[1].values)
        target_reals = target_reals.reshape(DURATION, HEIGHT, WIDTH)

        bucket_boundaries = torch.linspace(0, target_reals.max(), N_GUST_BUCKETS)
        targets = torch.bucketize(target_reals, bucket_boundaries)

        train_end_time = 34000
        if training:
            self._inputs = inputs[:train_end_time]
            self._targets = targets[:train_end_time]
        else:
            self._inputs = inputs[train_end_time:]
            self._targets = targets[train_end_time:]

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        return self._inputs[i], self._targets[i]

    def get_shift_factors(self) -> Tensor:
        return torch.min(self._inputs, dim=0)

    def get_scale_factors(self) -> Tensor:
        return self._inputs.std(0)

    def __len__(self) -> int:
        return self._inputs.size(0)


def load_field(name):
    if name == "wsgsmax":
        base_path = os.path.join(DATA_ROOT, "ukcp-other/wind-emulation-data")
    else:
        base_path = os.path.join(DATA_ROOT, "day/")
    pattern = os.path.join(base_path, f"{name}*.nc")
    dataset = xr.open_mfdataset(pattern)
    return dataset, dataset[name]

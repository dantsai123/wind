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


class PointwiseDataset(Dataset):
    def __init__(self, field_names: List[str], training: bool) -> None:
        self._n = DURATION
        self._d = len(field_names)

        input_data = [load_field(name)[1].values for name in field_names]
        input_tensors = [torch.tensor(d) for d in input_data]
        inputs = torch.stack(input_tensors, axis=4)
        inputs = inputs.reshape(DURATION, HEIGHT, WIDTH, self._d)
        targets = torch.tensor(load_field("wsgsmax")[1].values)
        targets = targets.reshape(DURATION, HEIGHT, WIDTH)

        train_times = range(0, 34000)

        self._inputs = inputs
        self._targets = targets

    def __get_item__(self, i: int) -> Tuple[Tensor, Tensor]:
        # lat, lon, time, val1, val2, ..., ...
        # time is just days since the first day
        # t =
        inputs = self._inputs.reshape(-1, self._d)[i]
        targets = self._targets.reshape(-1)[i]

    def get_shift_factors(self) -> List[float]:
        pass

    def get_scale_factors(self) -> List[float]:
        pass

    def __len__(self) -> int:
        pass


def load_field(name):
    if name == "wsgsmax":
        base_path = os.path.join(DATA_ROOT, "ukcp-other/wind-emulation-data")
    else:
        base_path = os.path.join(DATA_ROOT, "day/")
    pattern = os.path.join(base_path, f"{name}*.nc")
    dataset = xr.open_mfdataset(pattern)
    return dataset, dataset[name]

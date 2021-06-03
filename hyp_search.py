import itertools
from dataclasses import dataclass
from typing import List, Literal, Tuple

import GPy
import numpy as np
import ray
from GPy.core.mapping import Mapping
from GPy.kern import RBF, Linear
from GPy.mappings.constant import Constant
from GPy.mappings.kernel import Kernel
from GPy.models import GPRegression
from numpy import ndarray

import data
from data import DURATION, HEIGHT, WIDTH
from ray_tqdm import ray_tqdm


@dataclass
class FieldConfig:
    name: str
    kernel: Literal["rbf", "linear"]

    def __str__(self) -> str:
        return f"{self.name}_{self.kernel}"


@dataclass
class Config:
    fields: List[FieldConfig]
    mean: Literal["mean_gust", "constant"]

    def __str__(self) -> str:
        field_strs = [str(field) for field in self.fields]
        field_str = ",".join(field_strs)
        return f"{self.mean},{field_str}"


input_field_names = ["psl", "uas", "vas", "sfcWind", "pr"]


def main():
    ray.init()

    d = len(input_field_names)
    input_fields = [data.load_field(name)[1] for name in input_field_names]
    all_inputs = np.stack([f.values for f in input_fields], axis=4)
    all_inputs = all_inputs.reshape(DURATION, HEIGHT, WIDTH, d)

    _, wsgsmax_field = data.load_field("wsgsmax")
    all_wsgsmax = wsgsmax_field.values.reshape(DURATION, HEIGHT, WIDTH)

    train_start = 500
    train_length = 1080
    val_length = 1080
    test_length = 1080
    val_start = train_start + train_length
    test_start = val_start + val_length
    train_time = np.arange(train_start, train_start + train_length, 1)
    val_time = np.arange(val_start, val_start + val_length, 1)
    test_time = np.arange(test_start, test_start + test_length + test_length, 1)
    x = 55
    y = 40

    train_inputs = all_inputs[train_time, y, x, :]
    train_wsgsmax = all_wsgsmax[train_time, y, x]
    val_inputs = all_inputs[val_time, y, x, :]
    val_wsgsmax = all_wsgsmax[val_time, y, x]
    test_inputs = all_inputs[test_time, y, x, :]
    test_wsgsmax = all_wsgsmax[test_time, y, x]

    configs = []
    for mean in ["mean_gust", "constant"]:
        for fields in [["psl", "uas"], input_field_names[:4], input_field_names]:
            kernel_choices = itertools.combinations_with_replacement(
                ["rbf", "linear"], r=len(fields)
            )
            for kernels in kernel_choices:
                field_configs = [FieldConfig(f, k) for f, k in zip(fields, kernels)]
                configs.append(Config(field_configs, mean))

    print(f"Generated {len(configs)} configs")

    handles = [
        eval_config.remote(config, train_inputs, train_wsgsmax, val_inputs, val_wsgsmax)
        for config in configs
    ]
    ray_tqdm(handles)
    results = ray.get(handles)

    for config, result in sorted(zip(configs, results), key=lambda x: x[1][0]):
        print(f"{result[0]:.2f} {result[1]:.2f} {config}")


def field_d(name: str) -> int:
    """Returns the dimension in the input tensor containing the given field."""
    return input_field_names.index(name)


@ray.remote(num_cpus=4)
def eval_config(
    config: Config,
    train_inputs: ndarray,
    train_wsgsmax: ndarray,
    val_inputs: ndarray,
    val_wsgsmax: ndarray,
) -> Tuple[float, float]:
    if config.mean == "mean_gust":
        mean_add_gust = (train_wsgsmax - train_inputs[:, field_d("sfcWind")]).mean()
        mf = Mapping(train_inputs.shape[1], 1)
        mf.f = lambda x: (x[:, field_d("sfcWind")] + mean_add_gust).reshape(-1, 1)
        mf.update_gradients = lambda a, b: None
    elif config.mean == "constant":
        mf = Constant(train_inputs.shape[1], 1)
    else:
        raise NotImplementedError

    kernels = [create_kernel(field_config) for field_config in config.fields]
    kernel = kernels[0]
    for k in kernels[1:]:
        kernel = kernel * k

    model = GPRegression(
        train_inputs, train_wsgsmax.reshape(-1, 1), kernel, mean_function=mf
    )
    model.optimize(messages=True)
    return evaluate(model, val_inputs, val_wsgsmax)


def create_kernel(field_config: FieldConfig) -> Kernel:
    d = field_d(field_config.name)
    if field_config.kernel == "rbf":
        return RBF(input_dim=1, active_dims=[d])
    elif field_config.kernel == "linear":
        return Linear(input_dim=1, active_dims=[d])
    else:
        raise NotImplementedError


def mse(preds: ndarray, targets: ndarray) -> float:
    return ((preds.reshape(-1) - targets.reshape(-1)) ** 2).mean().item()


def evaluate(model: GPRegression, val_inputs: ndarray, val_wsgsmax: ndarray):
    # We consider MSE for all points, and MSE for "extreme" gusts
    # The hope is that considering the "extreme" gusts will help us hit the peaks
    extreme_gust_speed = 17
    means, vars = model.predict(val_inputs)

    all_mse = mse(means, val_wsgsmax)

    extreme_indices = val_wsgsmax > extreme_gust_speed
    extreme_mse = mse(means[extreme_indices], val_wsgsmax[extreme_indices])
    return all_mse, extreme_mse


if __name__ == "__main__":
    main()

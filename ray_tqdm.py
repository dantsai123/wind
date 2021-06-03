from typing import List
from tqdm import tqdm
import ray


def ray_tqdm(handles: List) -> None:
    progress_bar = tqdm(total=len(handles))
    awaiting = [handle for handle in handles]
    while True:
        ready, not_ready = ray.wait(awaiting)
        awaiting = not_ready
        progress_bar.update(len(ready))
        if len(awaiting) == 0:
            return

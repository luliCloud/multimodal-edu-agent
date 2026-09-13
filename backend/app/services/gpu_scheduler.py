"""In-process scheduling contract for the single-GPU prototype."""

from concurrent.futures import ThreadPoolExecutor
from threading import Condition
from typing import Callable, TypeVar

T = TypeVar("T")


class SingleGpuScheduler:
    def __init__(self, worker_count: int = 4) -> None:
        self.workers = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="video-worker")
        self._available = {0}
        self._condition = Condition()

    def run_on_gpu(self, task: Callable[[int], T]) -> T:
        # The slot allocator can later be backed by multiple GPU IDs.
        with self._condition:
            self._condition.wait_for(lambda: bool(self._available))
            gpu_id = min(self._available)
            self._available.remove(gpu_id)
        try:
            return task(gpu_id)
        finally:
            with self._condition:
                self._available.add(gpu_id)
                self._condition.notify()


scheduler = SingleGpuScheduler()

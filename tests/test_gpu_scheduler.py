import time
import unittest
from threading import Lock

from backend.app.services.gpu_scheduler import SingleGpuScheduler


class SingleGpuSchedulerTest(unittest.TestCase):
    def test_multiple_workers_serialize_gpu_tasks(self) -> None:
        scheduler = SingleGpuScheduler(worker_count=4)
        state = {"active": 0, "peak": 0}
        lock = Lock()

        def task(gpu_id: int) -> None:
            self.assertEqual(gpu_id, 0)
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.01)
            with lock:
                state["active"] -= 1

        futures = [scheduler.workers.submit(scheduler.run_on_gpu, task) for _ in range(8)]
        for future in futures:
            future.result(timeout=5)
        scheduler.workers.shutdown()
        self.assertEqual(state["peak"], 1)


if __name__ == "__main__":
    unittest.main()

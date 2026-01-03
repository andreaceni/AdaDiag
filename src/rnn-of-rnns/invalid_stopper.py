from typing import List
from ray.tune.stopper import Stopper


class InvalidStopper(Stopper):
    def __init__(self, metric: str, invalid_value: float):
        self.metric = metric
        self.invalid_value = invalid_value

    def __call__(self, trial_id, result):
        if result["train_loss"] == self.invalid_value or result["test_loss"] == self.invalid_value:
            return True
        return False

class NaNStopper(Stopper):
    def __init__(self, metrics: List[str]):
        self.metrics = metrics

    def __call__(self, trial_id, result):
        for metric in self.metrics:
            if result[metric] == float("nan"):
                return True
        return False
    
    def stop_all(self):
        return False
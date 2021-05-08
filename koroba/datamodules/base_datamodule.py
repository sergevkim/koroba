from abc import ABC, abstractmethod


class BaseDataModule(ABC):
    def __init__():
        pass

    @abstractmethod
    def setup():
        pass

    def get_true(self):
        return self.true

    def get_seen(self):
        return self.seen

    def get_optimized(self):
        return self.optimized


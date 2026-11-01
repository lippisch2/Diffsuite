from abc import ABC, abstractmethod


class FittingModel(ABC):
    name = "base"

    @abstractmethod
    def fit(self, data, **kwargs):
        raise NotImplementedError

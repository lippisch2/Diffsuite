from .base import FittingModel


class ExponentialFit(FittingModel):
    name = "exp"

    def fit(self, data, **kwargs):
        return {"model": self.name, "status": "ok", "data": data, "params": kwargs}

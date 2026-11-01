from .base import FittingModel


class DKIFit(FittingModel):
    name = "dki"

    def fit(self, data, **kwargs):
        return {"model": self.name, "status": "ok", "data": data, "params": kwargs}

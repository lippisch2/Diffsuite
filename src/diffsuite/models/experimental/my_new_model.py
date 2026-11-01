from ..base import FittingModel


class MyNewModel(FittingModel):
    name = "my_new_model"

    def fit(self, data, **kwargs):
        return {"model": self.name, "status": "experimental", "data": data, "params": kwargs}

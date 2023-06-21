from src.data_io.data_io import DataIO
from src.manager import Manager
from src.ml_model import MLModel


class ManagerManual(Manager):
    def __init__(
        self,
        data_io: DataIO,
        models: list[MLModel],
        params: dict,
        with_defects_only: bool = None,
        with_defects_only_clauses: list[str] = None,
        clean_all_nans: bool = False,
    ) -> None:
        super().__init__(
            data_io=data_io,
            models=models,
            with_defects_only=with_defects_only,
            with_defects_only_clauses=with_defects_only_clauses,
            clean_all_nans=clean_all_nans,
            params=params
        )

    def load_dates_train(self, panel_id):
        if (self.params["train_date_start"] is not None) & (
            self.params["train_date_end"] is not None
        ):
            return [self.params["train_date_start"], self.params["train_date_end"]]
        return super().load_dates_train(panel_id)

    def load_dates_predict(self, panel_id):
        if (self.params["predict_date_start"] is not None) & (
            self.params["predict_date_end"] is not None
        ):
            return [self.params["predict_date_start"], self.params["predict_date_end"]]
        return super().load_dates_predict(panel_id)

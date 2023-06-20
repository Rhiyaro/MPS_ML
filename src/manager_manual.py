from src.data_io.data_io import DataIO
from src.manager import Manager
from src.tchala import Tchala


class ManagerManual(Manager):
    def __init__(
        self,
        data_io: DataIO,
        tchalas: list[Tchala],
        params: dict,
        keep_local: bool = False,
        with_defects_only: bool = None,
        with_defects_only_clauses: list[str] = None,
        clean_all_nans: bool = False,
    ) -> None:
        super().__init__(
            data_io=data_io,
            tchalas=tchalas,
            keep_local=keep_local,
            with_defects_only=with_defects_only,
            with_defects_only_clauses=with_defects_only_clauses,
            clean_all_nans=clean_all_nans
        )
        self.params = params

    def load_dates_train(self, turbine_reg_id):
        if (self.params["train_date_start"] is not None) & (
            self.params["train_date_end"] is not None
        ):
            return [self.params["train_date_start"], self.params["train_date_end"]]
        return super().load_dates_train(turbine_reg_id)

    def load_dates_predict(self, turbine_reg_id):
        if (self.params["predict_date_start"] is not None) & (
            self.params["predict_date_end"] is not None
        ):
            return [self.params["predict_date_start"], self.params["predict_date_end"]]
        return super().load_dates_predict(turbine_reg_id)

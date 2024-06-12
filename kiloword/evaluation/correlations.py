import os
import pandas as pd
from .base import BaseTable


class CorrelationsTable(BaseTable):

    def __init__(self,
                 name: str = None,
                 table_folder: str = None,
                 table_columns: list = None):
        super().__init__(name, table_folder)
        self.table_struct = {col: [] for col in table_columns}
        self._load_table()

    def _load_table(self):
        self.table = None
        # The table exists: load the table directly
        if os.path.isfile(self.table_path):
            self.table = pd.read_csv(self.table_path)

        else:
            self.table = pd.DataFrame(self.table_struct)

    def update_table(self, data_dict_row: dict = None):
        if set(self.table.columns).intersection(set(data_dict_row.keys())) != set(self.table.columns):
            raise KeyError(f"Missing key(s) in data:  {set(self.table.columns) ^ set(data_dict_row.keys())}")
        row = [data_dict_row[k] for k in self.table.columns]
        self.table.loc[len(self.table.index)] = row

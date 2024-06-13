import os
import pandas as pd
from typing import Union
from .base import BaseTable


class CorrelationsTable(BaseTable):

    def __init__(self,
                 name: str = None,
                 table_folder: str = None,
                 table_columns: list = None,
                 eval: bool = False):

        self.table_struct = {col: [] for col in table_columns}
        self.eval = eval
        super().__init__(name, table_folder)
        self._load_table()

    def _load_table(self):
        self.table = None
        # The table exists: load the table directly
        if os.path.isfile(self.table_path):
            self.table = pd.read_csv(self.table_path)
        else:
            if self.eval:
                raise FileNotFoundError(f"Could not find the file {self.table_path}")
            self.table = pd.DataFrame(self.table_struct)

    def update_table(self, data_dict_row: dict = None):
        if set(self.table.columns).intersection(set(data_dict_row.keys())) != set(self.table.columns):
            raise KeyError(f"Missing key(s) in data:  {set(self.table.columns) ^ set(data_dict_row.keys())}")
        row = [data_dict_row[k] for k in self.table.columns]
        self.table.loc[len(self.table.index)] = row


    def extract_sub_table(self, attribute: str,
                          value: Union[str, float],
                          groupby_key: Union[str, list]=None):
        if attribute not in self.table.columns:
            raise KeyError(f"Key {attribute} is missing in the table")
        sub_table = self.table[self.table[attribute] == value]
        if groupby_key is not None:
            if groupby_key not in sub_table:
                raise KeyError(f"Key {groupby_key} is missing in the table")
            sub_table = sub_table.groupby(groupby_key)

        return sub_table

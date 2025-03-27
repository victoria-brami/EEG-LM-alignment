import numpy as np
import pandas as pd
import os

class BaseTable:

    def __init__(self,
                 name: str=None,
                 table_folder: str=None):
        self.table_folder = table_folder
        self.name = name
        self.table_path = os.path.join(self.table_folder, self.name)
        self._load_table()

    def _load_table(self):
        self.table = None
        # The table exists: load the table directly
        if os.path.isfile(self.table_path):
            self.table = pd.read_csv(self.table_path)

    def __len__(self):
        return len(self.table.index)

    def update_table(self):
        pass

    def save_table(self):
        self.table.to_csv(self.table_path, index=False)
        # print("Table saved to {}".format(self.table_path))
import pandas as pd
import numpy as np
from typing import Union

def extract_same_time_window_rows(window_start: int,
                                  window_end: int,
                                  tab: pd.DataFrame):
    extract_tab = tab[(tab["truncate_start"] == window_start) & (tab["truncate_end"] == window_end)]
    return extract_tab


def extract_same_electrode_rows(electrodes: Union[list, str],
                                  tab: pd.DataFrame):
    if isinstance(electrodes, list):
        extract_tab = tab[tab["Channels"].isin(electrodes)]
    else:
        extract_tab = tab[tab["Channels"] == electrodes]
    return extract_tab


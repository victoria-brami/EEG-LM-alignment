import mne

class Config:
    MNE_PATH = mne.datasets.kiloword.data_path(download=True)
    DATA = MNE_PATH / "kword_metadata-epo.fif"
    DATATXT = MNE_PATH / "KWORD_ERP_LEXICAL_DECISION_DGMH2015.txt"
    METADATA = MNE_PATH / "KWORD_VARIABLES_DGMH2015.txt"
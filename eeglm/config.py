import os


class Config:
    MNE_PATH = "./data"  # "/home/victoria-brami/Documents/personal/mne_data"  # mne.datasets.kiloword.data_path(download=True)
    LABELS_PATH = os.path.join(MNE_PATH, "words_and_pos.csv")
    SEMANTICS_PATH = os.path.join(MNE_PATH, "kiloword_semantic_labels.csv")
    DATA = os.path.join(MNE_PATH, "KWORD_ERP_LEXICAL_DECISION_DGMH2015.csv")
    DATATXT = os.path.join(MNE_PATH, "KWORD_ERP_LEXICAL_DECISION_DGMH2015.txt")
    METADATA = os.path.join(MNE_PATH, "KWORD_VARIABLES_DGMH2015.txt")
    LIST_LABELS = [
        "ENTERTAINMENT",
        "LOCATION",
        "SEPARATION",
        "MONEY",
        "NATURE",
        "QUANTITY",
        "FEELING",
        "MESSAGE",
        "POLITICS",
        "RELIGION",
        "HOUSE",
        "MOVE",
        "SPORT",
        "JUSTICE",
        "INDUSTRY",
        "LANGUAGE",
        "FOOD",
        "MODE",
        "DEVICE",
        "FAMILY",
        "MUSIC",
        "CRIME",
        "CATASTROPHE",
        "ARMY",
        "TIME",
        "SCHOOL",
        "CLEANNESS",
        "DEATH",
        "TRANSPORT",
        "TRAVEL",
        "DISTINCT",
        "GLORY",
        "BODY",
        "PEOPLE",
        "MEDICAL",
        "MATERIAL",
        "GOVERN",
        "SCIENCE",
        "PHILOSOPHY",
        "ABSTRACT",
    ]

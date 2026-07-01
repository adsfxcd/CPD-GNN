import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get(
    "CPD_GNN_DATA_DIR",
    os.path.join(PROJECT_ROOT, "features", "dataset", "dataset"),
)
PATH_TO_SAVE_ROOT = os.environ.get(
    "CPD_GNN_RESULT_DIR",
    os.path.join(PROJECT_ROOT, "result"),
)
os.makedirs(PATH_TO_SAVE_ROOT, exist_ok=True)

DATA_DIR_Win = {
    "CMUMOSI": os.path.join(DATA_ROOT, "CMUMOSI"),
    "CMUMOSEI": os.path.join(DATA_ROOT, "CMUMOSEI"),
    "IEMOCAPFour": os.path.join(DATA_ROOT, "IEMOCAPFour"),
    "IEMOCAPSix": os.path.join(DATA_ROOT, "IEMOCAP"),
}

PATH_TO_FEATURES_Win = {
    dataset: os.path.join(root, "features") for dataset, root in DATA_DIR_Win.items()
}

PATH_TO_LABEL_Win = {
    "CMUMOSI": os.path.join(DATA_DIR_Win["CMUMOSI"], "CMUMOSI_features_raw_2way.pkl"),
    "CMUMOSEI": os.path.join(DATA_DIR_Win["CMUMOSEI"], "CMUMOSEI_features_raw_2way.pkl"),
    "IEMOCAPSix": os.path.join(DATA_DIR_Win["IEMOCAPSix"], "IEMOCAP_features_raw_6way.pkl"),
    "IEMOCAPFour": os.path.join(DATA_DIR_Win["IEMOCAPFour"], "IEMOCAP_features_raw_4way.pkl"),
}

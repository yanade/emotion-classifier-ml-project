from pathlib import Path

HF_DATASET = "dair-ai/emotion"
DEFAULT_SPLIT = "train"
SPLITS = {
    "train": "split/train-00000-of-00001.parquet",
    "validation": "split/validation-00000-of-00001.parquet",
    "test": "split/test-00000-of-00001.parquet",
}

PATH_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = PATH_ROOT/"data"/"cleaned_dataset.csv"
TEXT_COL = "text"
LABEL_COL = "label"
LABEL_MAPPING = "emotion"

LABEL_MAPPING_PATH = PATH_ROOT/"data"/'label_mapping.json'

MODEL_PATH = PATH_ROOT/"src"/"model.pkl"

LANGUAGE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


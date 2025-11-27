import json
import logging
from datasets import load_dataset
from src.config import HF_DATASET, LABEL_MAPPING_PATH


def fetch_label_mapping():

    logging.info(f"Fetching label mapping for dataset: {HF_DATASET}")
    ds = load_dataset(HF_DATASET, split="train")
    class_names = ds.features["label"].names 
    mapping = {i: name for i, name in enumerate(class_names)}

    with open(LABEL_MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)

    logging.info(f"Label mapping saved to {LABEL_MAPPING_PATH}: {mapping}")
    return mapping



def load_label_mapping():
    try:
        with open(LABEL_MAPPING_PATH) as f:
            mapping = json.load(f)
        mapping = {int(k): v for k, v in mapping.items()}
        return mapping

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Label mapping not found at {LABEL_MAPPING_PATH}. "
        )
    

import pandas as pd
import logging
from config import TEXT_COL, LABEL_COL, HF_DATASET, DEFAULT_SPLIT, SPLITS, CLEAN_DATA_PATH, LABEL_MAPPING_PATH, LABEL_MAPPING
import json
from label_mapping import fetch_label_mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)



def load_data(split=DEFAULT_SPLIT):
    logging.info("Loading raw dataset ...")
    file_path = f"hf://datasets/{HF_DATASET}/" + SPLITS[split]
    df = pd.read_parquet(file_path)
    logging.info("Dataset loaded successfully.")
    return df


def clean_data(df):
    logging.info("Cleaning data...")
    df = df.drop_duplicates()
    df = df.dropna(subset=[TEXT_COL])
    df = df[df[TEXT_COL].str.strip() != ""]
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: " ".join(x.split()))
    df[LABEL_COL] = df[LABEL_COL].astype(int)    
    logging.info("Dataset cleaned  successfully.")
    return df
    

def save_data(df, CLEAN_DATA_PATH):
    df.to_csv(CLEAN_DATA_PATH, index=False)
    logging.info(f"Cleaned data saved to {CLEAN_DATA_PATH}")
    

    

if __name__ == "__main__":
    fetch_label_mapping()
    df_raw = load_data(DEFAULT_SPLIT)
    df_clean = clean_data(df_raw)
    save_data(df_clean, CLEAN_DATA_PATH)









import pandas as pd
from src.config import CLEAN_DATA_PATH, MODEL_PATH
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

def load_training_data(path):
    """Loads the cleaned dataset from the specified path."""
    logging.info(f"Loading cleaned data from {path}")
    return pd.read_csv(path)

def train_and_evaluate(df):
    """
    Builds and trains a classification pipeline, evaluates it,
    and returns the trained pipeline.
    """
    logging.info("Starting model training and evaluation...")
    
    X = df["text"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Define the steps of the pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2
        )),
        ('model', LogisticRegression(
            max_iter=300,
            solver='lbfgs',
            class_weight="balanced",
            C=1.0
        ))
    ])
    
    # Fit the entire pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    logging.info("\n--- Evaluation Metrics ---")
    logging.info("Accuracy: %.4f", accuracy)
    logging.info("Classification Report:\n%s", report)
    logging.info("Confusion Matrix:\n%s", cm)
    logging.info("--- End of Metrics ---")

    return pipeline

def save_pipeline(pipeline, path):
    """Saves the scikit-learn pipeline to the specified path."""
    logging.info(f"Saving model pipeline to {path}")
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logging.info("Pipeline saved successfully.")

def main():
    """Main function to run the training pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    df = load_training_data(CLEAN_DATA_PATH)
    pipeline = train_and_evaluate(df)
    save_pipeline(pipeline, MODEL_PATH)

if __name__ == "__main__":
    main()

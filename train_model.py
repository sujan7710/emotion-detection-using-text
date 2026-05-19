from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "emotion_dataset_expanded.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "emotion_model.joblib"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = {"text", "emotion"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    data = data.dropna(subset=["text", "emotion"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["emotion"] = data["emotion"].astype(str).str.strip().str.lower()
    data = data[data["text"] != ""]
    return data


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                stop_words="english",
                                ngram_range=(1, 3),
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                lowercase=True,
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                ComplementNB(alpha=0.08),
            ),
        ]
    )


def train_and_save_model() -> Pipeline:
    data = load_dataset()
    model = build_pipeline()

    class_counts = data["emotion"].value_counts()
    can_stratify = len(class_counts) > 1 and class_counts.min() >= 2

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["emotion"],
        test_size=0.2,
        random_state=42,
        stratify=data["emotion"] if can_stratify else None,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Model trained successfully")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_and_save_model()

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data


def train_models(X_train, y_train):

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=42
        ),

        "svm": SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        )
    }

    trained_models = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        trained_models[model_name] = model

        print(f"{model_name} training completed!")

    return trained_models


if __name__ == "__main__":

    file_path = (
        r"C:\Users\Tapaswini Shaw"
        r"\OneDrive\Desktop"
        r"\Ecommerce_Order_Cancellation_Prediction"
        r"\data\raw\df_train.csv"
    )

    # Load data
    df = load_data(file_path)

    # Preprocess data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data(df)

    # Train models
    trained_models = train_models(
        X_train,
        y_train
    )

    print("\nAll models trained successfully!")
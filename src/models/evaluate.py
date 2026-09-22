import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data


def evaluate_models(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=42
        ),

        "SVM": SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        )
    }

    results = {}

    for model_name, model in models.items():

        print("\n" + "=" * 50)
        print(f"Evaluating: {model_name}")
        print("=" * 50)

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Probabilities / decision scores
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_score
        )

        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(
            y_test,
            y_pred
        )

        print("\nConfusion Matrix:")
        print(cm)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Not Cancelled",
                "Cancelled"
            ]
        )

        display.plot()

        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":

    file_path = (
        r"C:\Users\Tapaswini Shaw"
        r"\OneDrive\Desktop"
        r"\Ecommerce_Order_Cancellation_Prediction"
        r"\data\raw\df_train.csv"
    )

    # Load dataset
    df = load_data(file_path)

    # Preprocess dataset
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data(df)

    # Evaluate models
    results = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON")
    print("=" * 50)

    for model_name, metrics in results.items():

        print(f"\n{model_name}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
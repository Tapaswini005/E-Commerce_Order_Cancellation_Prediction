import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(df):

    # --------------------------------------------------
    # 1. Create target variable
    # --------------------------------------------------

    df = df.copy()

    df["status"] = df["status"].map({
        "Not_Canceled": 0,
        "Canceled": 1
    })


    # --------------------------------------------------
    # 2. Remove unnecessary columns
    # --------------------------------------------------

    df = df.drop(columns=["id"])


    # --------------------------------------------------
    # 3. Convert slot_date into datetime
    # --------------------------------------------------

    df["slot_date"] = pd.to_datetime(
        df["slot_date"],
        errors="coerce"
    )


    # --------------------------------------------------
    # 4. Create date features
    # --------------------------------------------------

    df["slot_year"] = df["slot_date"].dt.year
    df["slot_month"] = df["slot_date"].dt.month
    df["slot_day"] = df["slot_date"].dt.day
    df["slot_day_of_week"] = df["slot_date"].dt.dayofweek
    df["is_weekend"] = (
        df["slot_day_of_week"] >= 5
    ).astype(int)


    # --------------------------------------------------
    # 5. Remove original date column
    # --------------------------------------------------

    df = df.drop(columns=["slot_date"])


    # --------------------------------------------------
    # 6. Separate features and target
    # --------------------------------------------------

    X = df.drop(columns=["status"])
    y = df["status"]


    # --------------------------------------------------
    # 7. Identify numerical and categorical columns
    # --------------------------------------------------

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()


    # --------------------------------------------------
    # 8. Numerical preprocessing
    # --------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )


    # --------------------------------------------------
    # 9. Categorical preprocessing
    # --------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


    # --------------------------------------------------
    # 10. Combine preprocessing
    # --------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


    # --------------------------------------------------
    # 11. Train/Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # --------------------------------------------------
    # 12. Fit preprocessing ONLY on training data
    # --------------------------------------------------

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)


    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":

    from load_data import load_data


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


    print("Preprocessing completed successfully!")

    print("\nOriginal dataset shape:")
    print(df.shape)

    print("\nTraining features shape:")
    print(X_train.shape)

    print("\nTesting features shape:")
    print(X_test.shape)

    print("\nTraining target shape:")
    print(y_train.shape)

    print("\nTesting target shape:")
    print(y_test.shape)

    print("\nTarget distribution:")
    print(y_train.value_counts())
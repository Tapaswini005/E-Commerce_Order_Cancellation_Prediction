import pandas as pd


def load_data(file_path):
    df = pd.read_csv(
        file_path,
        encoding="utf-8",
        low_memory=False
    )

    return df


if __name__ == "__main__":
    file_path = r"C:\Users\Tapaswini Shaw\OneDrive\Desktop\Ecommerce_Order_Cancellation_Prediction\data\raw\df_train.csv"

    df = load_data(file_path)

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())
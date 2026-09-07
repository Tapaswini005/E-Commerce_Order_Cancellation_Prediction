import pandas as pd


def load_data(file_path):
    return pd.read_csv('C:\Users\Tapaswini Shaw\OneDrive\Desktop\Ecommerce_Order_Cancellation_Prediction\data\raw\orders.csv')


if __name__ == "__main__":
    data = load_data("data/raw/orders.csv")

    print("Dataset loaded successfully!")
    print("Shape:", data.shape)
    print("\nColumns:")
    print(data.columns.tolist())
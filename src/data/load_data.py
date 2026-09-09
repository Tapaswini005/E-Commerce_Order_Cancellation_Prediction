import pandas as pd

def load_data(file_path):
    df = pd.read_csv(
        file_path,
        sep = ";",
        encoding = "utf-8",
        low_memory = False
    )
    
    return df

if __name__ == "__main__":
    file_path = r"C:\Users\Tapaswini Shaw\OneDrive\Desktop\Ecommerce_Order_Cancellation_Prediction\data\raw\orders.csv"
    
    df = load_data(file_path)
    
    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
       
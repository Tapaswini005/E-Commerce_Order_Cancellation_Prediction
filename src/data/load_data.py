import pandas as pd

df = pd.read_csv(
    r'C:\Users\Tapaswini Shaw\OneDrive\Desktop\Ecommerce_Order_Cancellation_Prediction\data\raw\orders.csv',
    encoding='utf-8',
    low_memory=False
)

print(df.head())
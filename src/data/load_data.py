import pandas as pd

df = pd.read_csv(
    r'C:\Users\Tapaswini Shaw\OneDrive\Desktop\Ecommerce_Order_Cancellation_Prediction\data\raw\orders.csv',
    sep=';',          # ✅ tells pandas to use semicolons
    encoding='utf-8', # or 'latin1' if utf-8 fails
    low_memory=False
)
print(df.head())
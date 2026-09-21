import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_data import load_data


def perform_eda(df):

    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Dataset shape
    # --------------------------------------------------

    print("\nDataset Shape:")
    print(df.shape)

    # --------------------------------------------------
    # 2. Dataset information
    # --------------------------------------------------

    print("\nDataset Information:")
    print(df.info())

    # --------------------------------------------------
    # 3. Missing values
    # --------------------------------------------------

    print("\nMissing Values:")
    print(df.isnull().sum())

    # --------------------------------------------------
    # 4. Duplicate rows
    # --------------------------------------------------

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    # --------------------------------------------------
    # 5. Target distribution
    # --------------------------------------------------

    print("\nOrder Status Distribution:")
    print(df["status"].value_counts())

    print("\nOrder Status Percentage:")
    print(
        df["status"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------
    # 6. Numerical summary
    # --------------------------------------------------

    print("\nNumerical Features Summary:")
    print(df.describe())

    # --------------------------------------------------
    # 7. Cancellation distribution
    # --------------------------------------------------

    plt.figure(figsize=(6, 4))

    df["status"].value_counts().plot(kind="bar")

    plt.title("Order Cancellation Distribution")
    plt.xlabel("Order Status")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 8. Cancellation by payment type
    # --------------------------------------------------

    payment_status = pd.crosstab(
        df["payment_type"],
        df["status"]
    )

    print("\nCancellation by Payment Type:")
    print(payment_status)

    payment_status.plot(kind="bar", figsize=(8, 5))

    plt.title("Order Status by Payment Type")
    plt.xlabel("Payment Type")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 9. Cancellation by app or website
    # --------------------------------------------------

    channel_status = pd.crosstab(
        df["app_or_website"],
        df["status"]
    )

    print("\nCancellation by Ordering Channel:")
    print(channel_status)

    channel_status.plot(kind="bar", figsize=(7, 5))

    plt.title("Order Status by App / Website")
    plt.xlabel("Ordering Channel")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 10. Cancellation by customer segment
    # --------------------------------------------------

    segment_status = pd.crosstab(
        df["customer_segmentation"],
        df["status"]
    )

    print("\nCancellation by Customer Segment:")
    print(segment_status)

    segment_status.plot(kind="bar", figsize=(8, 5))

    plt.title("Order Status by Customer Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 11. Lead time vs cancellation
    # --------------------------------------------------

    df.boxplot(
        column="lead_time",
        by="status",
        figsize=(7, 5)
    )

    plt.title("Lead Time by Order Status")
    plt.suptitle("")
    plt.xlabel("Order Status")
    plt.ylabel("Lead Time")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 12. Total price vs cancellation
    # --------------------------------------------------

    df.boxplot(
        column="total_price",
        by="status",
        figsize=(7, 5)
    )

    plt.title("Total Price by Order Status")
    plt.suptitle("")
    plt.xlabel("Order Status")
    plt.ylabel("Total Price")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    file_path = (
        r"C:\Users\Tapaswini Shaw"
        r"\OneDrive\Desktop"
        r"\Ecommerce_Order_Cancellation_Prediction"
        r"\data\raw\df_train.csv"
    )

    df = load_data(file_path)

    perform_eda(df)
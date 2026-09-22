import joblib
import pandas as pd


def load_model(model_name):
    model_path = f"models/{model_name}.pkl"
    return joblib.load(model_path)


def load_preprocessor():
    return joblib.load("models/preprocessor.pkl")


def get_user_input():

    print("\n" + "=" * 50)
    print("ORDER CANCELLATION PREDICTION")
    print("=" * 50)

    order = {}

    order["app_or_website"] = input(
        "App or Website: "
    )

    order["customer_collects"] = float(
        input("Customer Collects: ")
    )

    order["customer_segmentation"] = input(
        "Customer Segmentation: "
    )

    order["is_a_repeat_order"] = float(
        input("Is a Repeat Order: ")
    )

    order["lead_time"] = float(
        input("Lead Time: ")
    )

    order["n_customer_notes"] = int(
        input("Number of Customer Notes: ")
    )

    order["n_items_above_quantity_10"] = int(
        input("Items Above Quantity 10: ")
    )

    order["n_listed_addresses"] = int(
        input("Number of Listed Addresses: ")
    )

    order["n_listed_payment_methods"] = float(
        input("Number of Listed Payment Methods: ")
    )

    order["n_previous_cancelled_orders"] = int(
        input("Previous Cancelled Orders: ")
    )

    order["n_previous_completed_orders"] = int(
        input("Previous Completed Orders: ")
    )

    order["n_small_items"] = int(
        input("Number of Small Items: ")
    )

    order["payment_type"] = input(
        "Payment Type: "
    )

    order["slot_date"] = input(
        "Slot Date (YYYY-MM-DD): "
    )

    order["store_number"] = input(
        "Store Number: "
    )

    order["total_price"] = float(
        input("Total Price: ")
    )

    return order


def preprocess_input(order_data, preprocessor):

    df = pd.DataFrame([order_data])

    # Convert date
    df["slot_date"] = pd.to_datetime(
        df["slot_date"],
        errors="coerce"
    )

    # Create date features
    df["slot_year"] = df["slot_date"].dt.year

    df["slot_month"] = df["slot_date"].dt.month

    df["slot_day"] = df["slot_date"].dt.day

    df["slot_day_of_week"] = (
        df["slot_date"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["slot_day_of_week"] >= 5
    ).astype(int)

    # Remove original date
    df = df.drop(columns=["slot_date"])

    # Apply training preprocessing
    X = preprocessor.transform(df)

    return X


def predict_order(order_data, model_name):

    model = load_model(model_name)

    preprocessor = load_preprocessor()

    X = preprocess_input(
        order_data,
        preprocessor
    )

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    if prediction == 1:
        result = "Canceled"
    else:
        result = "Not_Canceled"

    return result, probability


if __name__ == "__main__":

    order = get_user_input()

    models = [
        "logistic_regression",
        "random_forest",
        "svm"
    ]

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    for model_name in models:

        result, probability = predict_order(
            order,
            model_name
        )

        print(f"\nModel: {model_name}")
        print(f"Prediction: {result}")
        print(
            f"Cancellation Probability: "
            f"{probability:.2%}"
        )

    print("\n" + "=" * 60)
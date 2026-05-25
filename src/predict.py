import os
import joblib
import pandas as pd


def load_model_package():
    model_package = joblib.load("models/model.joblib")

    print("Model package loaded successfully.")

    return model_package


def load_forward_data():
    X_forward_looking = pd.read_csv("data/X_forward_looking.csv")

    print("X_forward_looking loaded successfully.")
    print("X_forward_looking shape:", X_forward_looking.shape)

    return X_forward_looking


def encode_categorical_columns(X_forward_looking, label_encoders, categorical_columns):
    for col in categorical_columns:
        if col in X_forward_looking.columns and col in label_encoders:
            encoder = label_encoders[col]

            # Handle unseen labels safely
            known_classes = set(encoder.classes_)

            X_forward_looking[col] = X_forward_looking[col].astype(str).apply(
                lambda value: value if value in known_classes else encoder.classes_[0]
            )

            X_forward_looking[col] = encoder.transform(X_forward_looking[col].astype(str))

            print(f"Encoded column: {col}")

    return X_forward_looking


def drop_unnecessary_columns(X_forward_looking, drop_columns):
    for col in drop_columns:
        if col in X_forward_looking.columns:
            X_forward_looking = X_forward_looking.drop(columns=[col])

    return X_forward_looking


def keep_numeric_data(X_forward_looking):
    X_forward_looking = X_forward_looking.select_dtypes(include=["int64", "float64"])

    print("After keeping numeric columns:")
    print("X_forward_looking shape:", X_forward_looking.shape)

    return X_forward_looking


def impute_missing_values(X_forward_looking, imputer):
    X_forward_imputed = imputer.transform(X_forward_looking)

    X_forward_looking = pd.DataFrame(
        X_forward_imputed,
        columns=X_forward_looking.columns,
        index=X_forward_looking.index
    )

    print("Missing values handled.")

    return X_forward_looking


def select_saved_features(X_forward_looking, selected_columns):
    X_forward_selected = X_forward_looking.reindex(columns=selected_columns)

    print("Selected saved model features.")
    print("X_forward_selected shape:", X_forward_selected.shape)

    return X_forward_selected


def generate_predictions(model, X_forward_selected, target_columns):
    predictions = model.predict(X_forward_selected)

    prediction_df = pd.DataFrame(
        predictions,
        columns=target_columns
    )

    print("Predictions generated successfully.")
    print("Prediction shape:", prediction_df.shape)

    return prediction_df


def save_predictions(prediction_df):
    os.makedirs("data", exist_ok=True)

    prediction_df.to_csv("data/predictions.csv", index=False)

    print("Predictions saved successfully:")
    print("data/predictions.csv")


def main():
    model_package = load_model_package()

    model = model_package["model"]
    imputer = model_package["imputer"]
    label_encoders = model_package["label_encoders"]
    selected_columns = model_package["selected_columns"]
    target_columns = model_package["target_columns"]
    categorical_columns = model_package["categorical_columns"]
    drop_columns = model_package["drop_columns"]

    X_forward_looking = load_forward_data()

    X_forward_looking = encode_categorical_columns(
        X_forward_looking,
        label_encoders,
        categorical_columns
    )

    X_forward_looking = drop_unnecessary_columns(
        X_forward_looking,
        drop_columns
    )

    X_forward_looking = keep_numeric_data(X_forward_looking)

    X_forward_looking = impute_missing_values(
        X_forward_looking,
        imputer
    )

    X_forward_selected = select_saved_features(
        X_forward_looking,
        selected_columns
    )

    prediction_df = generate_predictions(
        model,
        X_forward_selected,
        target_columns
    )

    save_predictions(prediction_df)


if __name__ == "__main__":
    main()


    
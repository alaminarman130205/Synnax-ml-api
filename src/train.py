import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CATEGORICAL_COLUMNS = ["company_id", "country_code", "sector", "industry"]
DROP_COLUMNS = ["company_id"]

# Keep this small for poor PC performance
N_FEATURES = 50


def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_forward_looking = pd.read_csv("data/X_forward_looking.csv")
    targets_train = pd.read_csv("data/targets_train.csv")

    print("Data loaded successfully.")
    print("X_train shape:", X_train.shape)
    print("X_forward_looking shape:", X_forward_looking.shape)
    print("targets_train shape:", targets_train.shape)

    return X_train, X_forward_looking, targets_train


def encode_categorical_columns(X_train, X_forward_looking):
    label_encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col in X_train.columns and col in X_forward_looking.columns:
            encoder = LabelEncoder()

            combined_values = pd.concat(
                [
                    X_train[col].astype(str),
                    X_forward_looking[col].astype(str)
                ],
                axis=0
            )

            encoder.fit(combined_values)

            X_train[col] = encoder.transform(X_train[col].astype(str))
            X_forward_looking[col] = encoder.transform(X_forward_looking[col].astype(str))

            label_encoders[col] = encoder

            print(f"Encoded column: {col}")

    return X_train, X_forward_looking, label_encoders


def drop_unnecessary_columns(X_train, X_forward_looking, targets_train):
    for col in DROP_COLUMNS:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])

        if col in X_forward_looking.columns:
            X_forward_looking = X_forward_looking.drop(columns=[col])

        if col in targets_train.columns:
            targets_train = targets_train.drop(columns=[col])

    return X_train, X_forward_looking, targets_train


def keep_numeric_data(X_train, X_forward_looking, targets_train):
    X_train = X_train.select_dtypes(include=["int64", "float64"])
    X_forward_looking = X_forward_looking.select_dtypes(include=["int64", "float64"])
    targets_train = targets_train.select_dtypes(include=["int64", "float64"])

    print("\nAfter keeping numeric columns:")
    print("X_train shape:", X_train.shape)
    print("X_forward_looking shape:", X_forward_looking.shape)
    print("targets_train shape:", targets_train.shape)

    if targets_train.shape[1] == 0:
        raise ValueError("No numeric target columns found.")

    return X_train, X_forward_looking, targets_train


def impute_missing_values(X_train, X_forward_looking):
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)
    X_forward_imputed = imputer.transform(X_forward_looking)

    X_train = pd.DataFrame(
        X_train_imputed,
        columns=X_train.columns,
        index=X_train.index
    )

    X_forward_looking = pd.DataFrame(
        X_forward_imputed,
        columns=X_forward_looking.columns,
        index=X_forward_looking.index
    )

    print("\nMissing values handled with median imputer.")

    return X_train, X_forward_looking, imputer


def select_top_correlated_features(X_train, targets_train, n_features=N_FEATURES):
    """
    Fast feature selection:
    For each feature, calculate average absolute correlation with all target columns.
    Select top N features.
    """

    print("\nSelecting top correlated features...")

    scores = {}

    for feature in X_train.columns:
        feature_values = X_train[feature]

        target_correlations = []

        for target in targets_train.columns:
            corr = feature_values.corr(targets_train[target])

            if pd.notna(corr):
                target_correlations.append(abs(corr))

        if target_correlations:
            scores[feature] = np.mean(target_correlations)
        else:
            scores[feature] = 0

    sorted_features = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    selected_columns = sorted_features[:min(n_features, len(sorted_features))]

    X_train_selected = X_train[selected_columns]

    print(f"Selected top {len(selected_columns)} correlated features.")
    print("Selected features:")
    print(selected_columns)

    return X_train_selected, selected_columns


def train_linear_model(X_train, targets_train):
    print("\nTraining Linear Regression model...")

    model = MultiOutputRegressor(
        LinearRegression()
    )

    model.fit(X_train, targets_train)

    print("Linear Regression training completed.")

    return model


def evaluate_model(model, X_train, targets_train):
    predictions = model.predict(X_train)

    mae = mean_absolute_error(targets_train, predictions)
    mse = mean_squared_error(targets_train, predictions)
    r2 = r2_score(targets_train, predictions)

    print("\nTraining Evaluation:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 Score:", r2)


def save_model_package(
    model,
    imputer,
    label_encoders,
    selected_columns,
    target_columns
):
    os.makedirs("models", exist_ok=True)

    model_package = {
        "model_type": "linear_regression_top_correlation",
        "model": model,
        "imputer": imputer,
        "label_encoders": label_encoders,
        "selected_columns": selected_columns,
        "target_columns": target_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "drop_columns": DROP_COLUMNS
    }

    joblib.dump(model_package, "models/model.joblib")

    print("\nModel package saved successfully:")
    print("models/model.joblib")


def main():
    X_train, X_forward_looking, targets_train = load_data()

    X_train, X_forward_looking, label_encoders = encode_categorical_columns(
        X_train,
        X_forward_looking
    )

    X_train, X_forward_looking, targets_train = drop_unnecessary_columns(
        X_train,
        X_forward_looking,
        targets_train
    )

    X_train, X_forward_looking, targets_train = keep_numeric_data(
        X_train,
        X_forward_looking,
        targets_train
    )

    X_train, X_forward_looking, imputer = impute_missing_values(
        X_train,
        X_forward_looking
    )

    X_train_selected, selected_columns = select_top_correlated_features(
        X_train,
        targets_train,
        n_features=N_FEATURES
    )

    model = train_linear_model(
        X_train_selected,
        targets_train
    )

    evaluate_model(
        model,
        X_train_selected,
        targets_train
    )

    save_model_package(
        model=model,
        imputer=imputer,
        label_encoders=label_encoders,
        selected_columns=selected_columns,
        target_columns=targets_train.columns.tolist()
    )


if __name__ == "__main__":
    main()
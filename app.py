import os
import pandas as pd
from fastapi import FastAPI, HTTPException


app = FastAPI(
    title="Synnax ML Prediction API",
    description="API for returning ML prediction results to backend developer",
    version="1.0.0"
)


PREDICTION_FILE = "data/predictions.csv"


def load_predictions():
    """
    Load prediction CSV file.
    """

    if not os.path.exists(PREDICTION_FILE):
        raise FileNotFoundError(
            "predictions.csv not found. Please run: python src\\predict.py"
        )

    predictions = pd.read_csv(PREDICTION_FILE)

    return predictions


@app.get("/")
def home():
    return {
        "message": "Synnax ML Prediction API is running",
        "available_endpoints": [
            "/predictions",
            "/prediction/{row_id}"
        ]
    }


@app.get("/predictions")
def get_all_predictions():
    """
    Return all prediction results.
    """

    try:
        predictions = load_predictions()

        return {
            "total_predictions": len(predictions),
            "columns": predictions.columns.tolist(),
            "predictions": predictions.to_dict(orient="records")
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/prediction/{row_id}")
def get_prediction_by_row(row_id: int):
    """
    Return one prediction by row number.
    Example:
    /prediction/0
    /prediction/1
    /prediction/2
    """

    try:
        predictions = load_predictions()

        if row_id < 0 or row_id >= len(predictions):
            raise HTTPException(
                status_code=404,
                detail=f"Prediction row {row_id} not found."
            )

        prediction = predictions.iloc[row_id].to_dict()

        return {
            "row_id": row_id,
            "prediction": prediction
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
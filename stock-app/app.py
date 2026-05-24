from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import traceback

# ---------- LOAD MODEL & SCALER ----------
try:
    model = load_model("lstm_stock_model.h5")
    # get the time steps the model was trained with
    # typically shape: (None, n_steps, 1)
    n_steps = model.input_shape[1]
    print("Model loaded. n_steps =", n_steps)
except Exception as e:
    print("Error loading model:", e)
    raise

try:
    scaler = joblib.load("scaler.pkl")
    print("Scaler loaded.")
except Exception as e:
    print("Error loading scaler:", e)
    raise


# ---------- FASTAPI SETUP ----------
app = FastAPI()


class PredictRequest(BaseModel):
    last_100_closes: list[float]  # can be more than n_steps


class PredictResponse(BaseModel):
    predicted_prices: list[float]


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        closes = req.last_100_closes

        # ensure enough points
        if len(closes) < n_steps:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {n_steps} closing prices, got {len(closes)}",
            )

        # use only the last n_steps values
        closes = closes[-n_steps:]

        # scale
        data = np.array(closes).reshape(-1, 1)         # (n_steps, 1)
        data_scaled = scaler.transform(data)

        # reshape for LSTM
        x_input = data_scaled.reshape(1, n_steps, 1)   # (1, n_steps, 1)
        temp_input = list(x_input.flatten())
        lst_output = []

        # predict next 30 steps
        for _ in range(30):
            x = np.array(temp_input[-n_steps:]).reshape(1, n_steps, 1)
            yhat = model.predict(x, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])

        # inverse scale
        preds_scaled = np.array(lst_output).reshape(-1, 1)
        preds = scaler.inverse_transform(preds_scaled).flatten().tolist()

        return PredictResponse(predicted_prices=preds)

    except HTTPException:
        raise
    except Exception as e:
        print("Error in /predict:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

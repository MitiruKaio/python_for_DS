from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()

# Загрузка модели и scaler
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)


class PredictionInput(BaseModel):
    lat: float = 55.55
    lon: float = 37.37
    total_square: int = 50
    floor: int = 3
    parsed_rooms: int = 2
    source: str = "ЦИАН"


# liveness-проба
@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)


# Функция для предсказаний
def make_prediction(lat, lon, total_square, floor, parsed_rooms, source):
    try:
        input_data = pd.DataFrame([{
            "lat": lat,
            "lon": lon,
            "total_square": total_square,
            "floor": floor,
            "parsed_rooms": parsed_rooms,
            "source": source
        }])

        input_data_encoded = pd.get_dummies(input_data, columns=['source'], prefix='source')

        feature_columns = np.load('feature_columns2.npy', allow_pickle=True).tolist()

        for col in feature_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0

        input_data_encoded = input_data_encoded[feature_columns]

        scaled_data = scaler.transform(input_data_encoded)

        prediction = model.predict(scaled_data)
        return {"prediction": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# GET-запрос
@app.get("/predict_get")
def predict_get(
        lat: float = 55.55,
        lon: float = 37.37,
        total_square: int = 50,
        floor: int = 3,
        parsed_rooms: int = 2,
        source: str = "ЦИАН"
):
    return make_prediction(lat, lon, total_square, floor, parsed_rooms, source)


# POST-запрос
@app.post("/predict_post")
def predict_post(data: PredictionInput):
    return make_prediction(**data.dict())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

import joblib  # Используется для загрузки модели
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional
import os
import data_preprocessing

# Инициализация приложения FastAPI
app = FastAPI()

# Загрузка модели
model = joblib.load("model.pkl")


# Определение класса для одного объекта
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

@app.get("/")
def read_root():
    return {"message": "API is working!"}


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    try:
        # Считываем передаваемый объект и преобразуем его в датафрейм        
        data = pd.DataFrame([item.model_dump()])
        # Делаем предварительную обработку данных, вынесенную в файл cuxtom_pipeline.py
        data = data_preprocessing.extract_numeric_from_cols(data)
        # Выполняем предсказание
        prediction = model.predict(data)
        return prediction
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка при предсказании: {e}")

# Изменил предложенный в задании шаблон для того, чтобы передавать в метод не коллекцию items, а .csv файл, как требовалось в задании
@app.post("/predict_csv/")
# На случай, если передаваемый .csv файл очень большой, сделаем функцию асинхронной
async def predict_csv(file: UploadFile):
    try:
        contents = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        data = pd.read_csv(temp_path)
        os.remove(temp_path)
        data = data_preprocessing.extract_numeric_from_cols(data)
        predictions = model.predict(data)
        data["predictions"] = predictions

        result_path = "prediction.csv"
        data.to_csv(result_path, index=False)
        return {"message": "Предсказания выполнены", "prediction_file": result_path}
    except Exception as e:
        print(f"Ошибка в predict_csv: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка при обработке файла: {e}")

# uvicorn app:app --reload

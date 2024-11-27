import joblib  # Используется для загрузки модели
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import os

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

# Определение класса для коллекции объектов
class Items(BaseModel):
    objects: List[Item]
    

@app.get("/")
def read_root():
    return {"message": "API is working!"}


# Метод для обработки одного объекта
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    try:
        # Преобразование входных данных в датафрейм
        data = pd.DataFrame([item.model_dump()])
        prediction = model.predict(data)[0]  # Получение предсказания
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при предсказании: {e}")

# Метод для обработки CSV-файлов
@app.post("/predict_csv/")
async def predict_csv(file: UploadFile):
    try:
        # Чтение CSV-файла
        contents = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        data = pd.read_csv(temp_path)
        os.remove(temp_path)  # Удалить временный файл после загрузки данных

        # Предсказания
        predictions = model.predict(data)
        data["predictions"] = predictions

        # Сохранение результатов в новый CSV-файл
        result_path = "prediction.csv"
        data.to_csv(result_path, index=False)

        return {"message": "Предсказания выполнены", "predictio_file": result_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при обработке файла: {e}")
    

# uvicorn app:app --reload

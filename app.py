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

# Фактически сейчас я пытаюсь загрузить в мою модель данные в следующем формате:
# {
#     "year": 2010.0,
#     "km_driven": 168000.0,
#     "mileage": 14.0,
#     "engine": 2498.0,
#     "max_power": 112.0,
#     "torque": 260.0,
#     "name_Honda City": 0.0,
#     "name_Hyundai Grand": 0.0,
#     "name_Hyundai Santro": 0.0,
#     "name_Hyundai Verna": 0.0,
#     "name_Hyundai i10": 0.0,
#     "name_Hyundai i20": 0.0,
#     "name_Mahindra Bolero": 0.0,
#     "name_Mahindra Scorpio": 0.0,
#     "name_Mahindra XUV500": 0.0,
#     "name_Maruti Alto": 0.0,
#     "name_Maruti Ertiga": 0.0,
#     "name_Maruti Swift": 0.0,
#     "name_Maruti Wagon": 0.0,
#     "name_Tata Indica": 0.0,
#     "name_Tata Indigo": 0.0,
#     "name_Toyota Innova": 0.0,
#     "name_rare": 0.0,
#     "fuel_Diesel": 1.0,
#     "fuel_LPG": 0.0,
#     "fuel_Petrol": 0.0,
#     "seller_type_Individual": 1.0,
#     "seller_type_Trustmark Dealer": 0.0,
#     "transmission_Manual": 1.0,
#     "owner_Fourth & Above Owner": 0.0,
#     "owner_Second Owner": 0.0,
#     "owner_Test Drive Car": 0.0,
#     "owner_Third Owner": 0.0,
#     "seats_4": 0.0,
#     "seats_5": 0.0,
#     "seats_6": 0.0,
#     "seats_7": 1.0,
#     "seats_8": 0.0,
#     "seats_9": 0.0,
#     "seats_10": 0.0,
#     "seats_14": 0.0
# }
# Потому что моя модель обучена именно на таких данных. Я понимаю, что нужно упаковать все преобразования данных в один пайплайн
# и применять его и к обучающим и к тестовым данным, но у меня не получилось так сделать. Файл с неработающим пайпланом
# лежит в репозитории

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

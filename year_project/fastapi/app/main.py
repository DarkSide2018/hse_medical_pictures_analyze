from fastapi import FastAPI, File, UploadFile
import time
import os
from functions import get_list_of_models, get_labels, get_label
import imghdr
import pickle

# import redis
# redis_client = redis.Redis(host=config.host.get_secret_value(), port=6379, db=0)
# from config_reader import config

from tgb_service import (
    create_predictable_dataframe,
    create_predictable_dataframe_cat_boost,
)

app = FastAPI()


@app.get("/")
def read_root():
    return {
        """use next commands:
        1. (get)ping - to see that service is working;
        2. (get)models - to get list of existing models;
        3. (get)predict/model/(base64_array) - to get prediction by picture;
        4. (get)info - to see information about the project
        5. (get)labels - to see all existing labels of the dataset
        """
    }


@app.post("/predict/{model}")
async def get_prediction(model: str, file: UploadFile = File(...)):
    # 1. check that there is such model
    models = get_list_of_models("models")
    if model not in models:
        return {
            "message": "There is no such model in service. Use /models to see the list of existing models"
        }

    # 2. Receive picture
    with open(os.path.join("saved_pics/", file.filename), "wb") as buffer:
        buffer.write(await file.read())

    # 3. Checking what we have received
    downloaded_file = "saved_pics/" + file.filename

    image_type = imghdr.what(downloaded_file)

    if image_type != "jpeg" and image_type != "jpg":
        return {"message": "Incorrect picture!"}

    # 4. Return label
    file_path = f"models/{model}.pickle"

    pickled_model = pickle.load(open(file_path, "rb"))

    if model == "cat_boost":
        dataframe = create_predictable_dataframe_cat_boost(downloaded_file)
    else:
        dataframe = create_predictable_dataframe(downloaded_file)

    predicted = pickled_model.predict(dataframe)

    return {"Predicted Label": get_label(predicted[0][0])}


@app.get("/ping")
async def ping():
    start_time = time.time()
    elapsed_time = time.time() - start_time
    return {"message": "pong", "response_time": elapsed_time}


@app.get("/models")
async def provide_models():
    models = get_list_of_models("models")
    if len(models) > 1:
        return {"message": f"There are {len(models)} working models", "models": models}
    else:
        return {"message": "There are no working models. Contant tech support!"}


@app.get("/labels")
async def provide_labels():
    labels = get_labels()
    if len(labels) > 1:
        return {"message": f"There are {len(labels)} labels", "labels": labels}
    else:
        return {"message": "Something wrong with the service. Contant tech support!"}


@app.get("/info")
async def show_info():
    return {
        "message": """
            ## Тема: обработка медицинских изображений(27)

            ## Датасет 
            https://www.kaggle.com/datasets/shubhamgoel27/dermnet

            ### Описание датасета
            Датасет представляет собой 19500 изображений разбитых на обучающие и тестовые наборы (15500 / 4000).
            Данные состоят из изображений 23 типов кожных заболеваний, взятых с http://www.dermnet.com/dermatology-pictures-skin-disease-pictures. 
            Категории включают прыщи, меланому, экзему, себорейный кератоз, стригущий лишай, буллезную болезнь, ядовитый плющ, псориаз, сосудистые опухоли и т. д.

            # Описание

            Цель данного проекта - создать рабочую CV модель для классификации изображений к одной из 23 категории заболеваний.
            А так же создать сервис для загрузки и предобработки изображений.
            Дополнительно в рамках проекта будут проведены работы по созданию ci/cd - автоматизации процесса сборки, настройки и развертывания сервиса и модели.   
            """
    }

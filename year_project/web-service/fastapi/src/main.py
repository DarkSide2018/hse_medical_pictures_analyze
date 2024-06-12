import time
import numpy as np

from fastapi import FastAPI, Path, UploadFile
from typing import Annotated
from datetime import datetime

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis

from torch import load, device
from torch.nn import Linear
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import efficientnet_v2_s


app = FastAPI(title='HSE year project app')

# Дата и время старта работы сервиса
started_at = datetime.now().strftime("%Y-%m-%d %H:%M")

# Список возможных болезней (на них обучалась модель)
diseases = {0: 'Light Diseases and Disorders of Pigmentation', 1: 'Lupus and other Connective Tissue diseases',
            2: 'Acne and Rosacea Photos', 3: 'Systemic Disease', 4: 'Poison Ivy Photos and other Contact Dermatitis',
            5: 'Vascular Tumors', 6: 'Urticaria Hives', 7: 'Atopic Dermatitis Photos', 8: 'Bullous Disease Photos',
            9: 'Hair Loss Photos Alopecia and other Hair Diseases', 10: 'Tinea Ringworm Candidiasis and other Fungal Infections',
            11: 'Psoriasis pictures Lichen Planus and related diseases', 12: 'Melanoma Skin Cancer Nevi and Moles',
            13: 'Nail Fungus and other Nail Disease', 14: 'Scabies Lyme Disease and other Infestations and Bites',
            15: 'Eczema Photos', 16: 'Exanthems and Drug Eruptions', 17: 'Herpes HPV and other STDs Photos',
            18: 'Seborrheic Keratoses and other Benign Tumors', 19: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
            20: 'Vasculitis Photos', 21: 'Cellulitis Impetigo and other Bacterial Infections', 22: 'Warts Molluscum and other Viral Infections'}

# Количество загруженных для предиктов изображений (для статистики)
images_loaded = 0

# Список с оценками пользователей
review = []

# Модель
model = efficientnet_v2_s()
model.classifier[1] = Linear(1280, 23)
model.load_state_dict(load('EfficientNetV2-S.pth', map_location=device('cpu')))
model.eval()

# Предобработка изображения
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# >>> Хэндлеры <<<

# Корневая директория
@app.get('/')
def root():
    return {'status': 'successful',
            'message': 'Hello! This is a web-site for classifying images with '
                       'skin diseases using deep learning models.'}


# Получение списка всех болезней
@app.get('/diseases/all')
def get_diseases() -> dict:
    if diseases:
        return {'status': 'successful',
                'data': diseases}
    else:
        return {'status': 'failed',
                'message': 'The list of diseases is empty!'}


# Получение названия болезни по его ID
@app.get('/diseases/{disease_id}')
@cache(expire=30)
def get_disease_name(disease_id: int) -> dict:
    time.sleep(2)
    if disease_id in diseases:
        return {'status': 'successful',
                'data': diseases[disease_id]}
    else:
        return {'status': 'failed',
                'message': f'There is no disease with disease_id {disease_id}!'}


# Добавление новой болезни
@app.post('/diseases/new')
def post_new_disease(disease_id: int, disease_name: str) -> dict:
    if disease_id in diseases:
        return {'status': 'failed',
                'message': f'Disease_id {disease_id} is busy!'}
    else:
        diseases[disease_id] = disease_name
        return {'status': 'successful',
                'message': f'Disease {disease_name} with disease_id {disease_id} has been added!'}


# Получение предсказания для загруженного изображения
@app.post('/predict')
def predict(file_uploaded: UploadFile) -> dict:
    file_location = f"tmp_files/{file_uploaded.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file_uploaded.file.read())

    image = read_image(file_location) / 255
    image = preprocess(image).unsqueeze(0)
    prediction = model(image)

    global images_loaded
    images_loaded += 1  # без global не работает

    return {'status': 'successful',
            'filename': file_uploaded.filename,
            'predict': f'Your disease is {diseases[prediction.argmax(1).item()]}'}


# Получение статистики о работе сервиса
@app.get('/stats')
def get_stats() -> dict:
    return {'status': 'successful',
            'data': {'started_at': started_at,
                     'images_loaded': images_loaded,
                     'rating': round(np.mean(review), 2) if review else 'There are no reviews yet!'}}


# Оставить отзыв о работе сервиса
@app.post('/review/{rating}')
def post_review(rating: Annotated[int, Path(ge=1, le=5)]) -> dict:
    review.append(rating)
    return {'status': 'successful',
            'message': f'Your mark {rating} has been added. Thank you!'}


# Подключаем Redis
@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost:5370")  # 83.222.9.144
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

import os
from aiogram import types


def get_list_of_models(path):
    models = []
    try:
        for model in os.listdir(path):
            models.append(model.split(".")[0])
    except Exception as e:
        print(e)
    return models


def show_models(path):
    models = get_list_of_models(path)
    kb = []
    if len(models) > 0:
        for model in models:
            kb.append([types.KeyboardButton(text=model)])
    else:
        kb.append([types.KeyboardButton(text="Обратитесь к системному администратору")])
    return types.ReplyKeyboardMarkup(
        keyboard=kb, resize_keyboard=True, input_field_placeholder="Выберите модель"
    )


def save_choice(model, path=""):
    if not model:
        model = "xgb"
    with open(f"{path}user_choices.txt", "w") as file:
        file.write(model)


def get_choice(path=""):
    model = ""
    with open(f"{path}user_choices.txt", "r") as file:
        model = file.read()
    return model if model else "xgb"


def get_label(label_number):
    labels = [
        "Acne and Rosacea Photos",
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
        "Atopic Dermatitis Photos",
        "Bullous Disease Photos",
        "Cellulitis Impetigo and other Bacterial Infections",
        "Eczema Photos",
        "Exanthems and Drug Eruptions",
        "Hair Loss Photos Alopecia and other Hair Diseases",
        "Herpes HPV and other STDs Photos",
        "Light Diseases and Disorders of Pigmentation",
        "Lupus and other Connective Tissue diseases",
        "Melanoma Skin Cancer Nevi and Moles",
        "Nail Fungus and other Nail Disease",
        "Poison Ivy Photos and other Contact Dermatitis",
        "Psoriasis pictures Lichen Planus and related diseases",
        "Scabies Lyme Disease and other Infestations and Bites",
        "Seborrheic Keratoses and other Benign Tumors",
        "Systemic Disease",
        "Tinea Ringworm Candidiasis and other Fungal Infections",
        "Urticaria Hives",
        "Vascular Tumors",
        "Vasculitis Photos",
        "Warts Molluscum and other Viral Infections",
    ]
    return labels[label_number] if label_number < len(labels) else "Moder Error"

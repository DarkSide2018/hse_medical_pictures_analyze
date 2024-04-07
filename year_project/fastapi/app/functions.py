import os


def get_list_of_models(path):
    models = []
    try:
        for model in os.listdir(path):
            models.append(model.split(".")[0])
    except Exception as e:
        print(e)
    return models


def get_labels():
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
    return labels


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

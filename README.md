# HSE МОВС23 - годовой проект

![Alt text](assets/image.png)

## Тема: обработка медицинских изображений(27)

## Датасет 
https://www.kaggle.com/datasets/shubhamgoel27/dermnet

**[Датасет на Yandex диске](https://disk.yandex.com/d/kNMbUOPlEyVXkA)**
### Описание датасета
Датасет представляет собой 19500 изображений разбитых на обучающие и тестовые наборы (15500 / 4000).

Данные состоят из изображений 23 типов кожных заболеваний, взятых с http://www.dermnet.com/dermatology-pictures-skin-disease-pictures. 

Категории включают прыщи, меланому, экзему, себорейный кератоз, стригущий лишай, буллезную болезнь, ядовитый плющ, псориаз, сосудистые опухоли и т. д.



### Список категорий

|    #   |          Название категории              | Обучающая/тестовая выборки |
|-------|-------------------------------------------|--------------------------|
|   1   | Acne and Rosacea Photos                   | 840 / 312                |
|   2   | Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions | 1149 / 288 |
|   3   | Atopic Dermatitis Photos                  | 489 / 123                |
|   4   | Bullous Disease Photos                    | 448 / 113                |
|   5   | Cellulitis Impetigo and other Bacterial Infections | 288 / 73         |
|   6   | Eczema Photos                             | 1235 / 309               |
|   7   | Exanthems and Drug Eruptions               | 404 / 101                |
|   8   | Hair Loss Photos Alopecia and other Hair Diseases | 239 / 60          |
|   9   | Herpes HPV and other STDs Photos           | 405 / 102                |
|  10   | Light Diseases and Disorders of Pigmentation | 568 / 143              |
|  11   | Lupus and other Connective Tissue diseases | 420 / 105                |
|  12   | Melanoma Skin Cancer Nevi and Moles        | 463 / 116                |
|  13   | Nail Fungus and other Nail Disease         | 1040 / 261               |
|  14   | Poison Ivy Photos and other Contact Dermatitis | 260 / 65             |
|  15   | Psoriasis pictures Lichen Planus and related diseases | 1405 / 352      |
|  16   | Scabies Lyme Disease and other Infestations and Bites | 431 / 108      |
|  17   | Seborrheic Keratoses and other Benign Tumors | 1371 / 108            |
|  18   | Systemic Disease                          | 606 / 152                |
|  19   | Tinea Ringworm Candidiasis and other Fungal Infections | 1300 / 325        |
|  20   | Urticaria Hives                           | 212 / 53                 |
|  21   | Vascular Tumors                           | 482 / 121                |
|  22   | Vasculitis Photos                         | 416 / 105                |
|  23   | Warts Molluscum and other Viral Infections | 1086 / 272               |


## Команда:
+ **Кирилл Филатов**
+ **Роман Гапонов**
+ **Роман Залесинский**

**Куратор: Кирилл Козлов**

# Описание

Цель данного проекта - создать рабочую CV модель для классификации изображений к одной из 23 категории заболеваний.

А так же создать сервис для загрузки и предобработки изображений.

Дополнительно в рамках проекта будут проведены работы по созданию ci/cd - автоматизации процесса сборки, настройки и развертывания сервиса и модели.   

# Как запустить проект
- Склонировать репозиторий

> git clone https://github.com/DarkSide2018/hse_medical_pictures_analyze.git

- Создать виртуальное окружение

> virtualenv venv

> source venv/bin/activate

- Установить зависимости:

> poetry install




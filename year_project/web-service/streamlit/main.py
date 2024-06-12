# Веб-интерфейс для сервиса по распознования кожных заболеваний

import streamlit as st
import requests
import time


url_root = 'http://83.222.9.144:9999'

st.set_page_config(layout='wide')
st.title('Detecting skin diseases using Deep Learning')

col1, col2 = st.columns(2)

with col1:
    # Загружаем изображение и получаем предсказание
    st.write('### Prediction :syringe:')
    uploaded_file = st.file_uploader('Upload your image:', type=['jpg', 'png', 'jpeg'],
                                     help='You need to upload an image that you need to make a prediction for. '
                                          'You can use the following file formats: `jpg`, `png` or `jpeg`')

    if st.button("Get prediction", type="primary"):
        if uploaded_file is not None:
            with st.spinner("Loading..."):
                time.sleep(0.5)
                url = url_root + '/predict'
                files = {'file_uploaded': uploaded_file}
                response = requests.post(url, files=files)
            st.success(f'Done! {response.json()["predict"]}')
        else:
            st.warning('Choose your image, please!', icon='😠')

    if st.button("Reset", type="secondary"):
        uploaded_file = None

    # Информаиця об используемой модели
    st.write('### What model is being used? :magic_wand:')
    st.write('Neural Network with architecture `EfficientNetV2` is used for predictions. This model is based on the '
             '[EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).')
    st.write('Compared to other popular architectures for CV, it has better results:')
    st.image('test-metrics.png')

with col2:
    # Информация о датасете
    st.write('### About Dataset')
    st.write('The data consists of images of 23 types of skin diseases taken from '
             'https://www.kaggle.com/datasets/shubhamgoel27/dermnet. '
             'The total number of images are around 19.500, out of which approximately '
             '15.500 have been split in the training set and the remaining in the test set.')
    st.write('The images are taken from the public portal Dermnet (https://dermnet.com/) '
             'which is the largest dermatology source online built for the purpose of providing '
             'online medical education.')
    st.write('Samples have the following class distributions:')
    st.image('distributions.jpg')
    st.write('Class imbalance is corrected by adding augmentations to the training dataset.')

# Всплывающий Sidebar слева
with st.sidebar:
    # Название окна
    st.title('HSE: year project')

    # Информация о сервисе
    url = url_root + '/stats'
    response = requests.get(url)
    st.write(f"The service has been running since {response.json()['data']['started_at'].split()[0]} and has "
             f"{response.json()['data']['images_loaded']} predicted diseases. The average rating "
             f"of the service is {response.json()['data']['rating']}.")

    # Информаиця о разработчикам сервиса
    st.markdown(
        """
        Service developers :
        - Kirill Filatov
        - Roman Gaponov
        - Roman Zalesinskiy
        """)

    # Оставить отзыв сервису
    rating = st.slider('Leave a review about the service, please!', 1, 5, 5, 1)
    url = url_root + f'/review/{rating}'
    response = requests.post(url)
    st.success(response.json()['message'])

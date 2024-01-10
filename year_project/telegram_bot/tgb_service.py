import io
import os

import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from year_project.telegram_bot.extracting import round_half_up, get_hog_mean, get_hog_std, get_harris_corners_count, \
    get_harris_corner_mean, calculate_channel_average_v2, count_hough_circles
from year_project.telegram_bot.functions import get_connection

load_dotenv()

BOT_TOKEN = os.environ.get('TOKEN')
chat_id = os.environ.get('chat_id')

print(BOT_TOKEN)

url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={chat_id}'


def create_predictable_dataframe_cat_boost(image):
    read = image.read()
    red_channel_intensity = round_half_up(calculate_channel_average_v2(read, "R"), 5)
    blue_channel_intensity = round_half_up(calculate_channel_average_v2(read, "B"), 5)
    green_channel_intensity = round_half_up(calculate_channel_average_v2(read, "G"), 5)
    img = Image.open(io.BytesIO(read))
    hog_mean = round_half_up(get_hog_mean(img), 5)
    hog_std = round_half_up(get_hog_std(img), 5)
    hough_circle = count_hough_circles(np.array(img))
    new_row = {
        'red_channel_intensity': red_channel_intensity,
        'blue_channel_intensity': blue_channel_intensity,
        'green_channel_intensity': green_channel_intensity,
        'HOG_mean': hog_mean,
        "houghCircle": hough_circle,
        'harris_count': get_harris_corners_count(img=np.array(img)),
        'harris_count_mean': get_harris_corner_mean(img=np.array(img)),
        'HOG_std': hog_std
    }
    data_frame = pd.DataFrame.from_dict(data=new_row, orient='index').T
    for i in range(2):
        data_frame = pd.concat([data_frame, data_frame])
    print(data_frame)
    scaler = StandardScaler()
    dataframe_train_scaled = scaler.fit_transform(data_frame)
    pca = PCA(n_components=4)
    return pca.fit_transform(dataframe_train_scaled)

def create_predictable_dataframe(image):
    read = image.read()
    red_channel_intensity = round_half_up(calculate_channel_average_v2(read, "R"), 5)
    blue_channel_intensity = round_half_up(calculate_channel_average_v2(read, "B"), 5)
    green_channel_intensity = round_half_up(calculate_channel_average_v2(read, "G"), 5)
    img = Image.open(io.BytesIO(read))
    hog_mean = round_half_up(get_hog_mean(img), 5)
    hog_std = round_half_up(get_hog_std(img), 5)
    new_row = {
        'red_channel_intensity': red_channel_intensity,
        'blue_channel_intensity': blue_channel_intensity,
        'green_channel_intensity': green_channel_intensity,
        'HOG_mean': hog_mean,
        'harris_count': get_harris_corners_count(img=np.array(img)),
        'harris_count_mean': get_harris_corner_mean(img=np.array(img)),
        'HOG_std': hog_std
    }
    return pd.DataFrame.from_dict(data=new_row, orient='index').T



def get_label(number):
    cursor = get_connection().cursor()
    sql1 = f'''select label from target_dictionary where target = {number}'''
    cursor.execute(sql1)
    data_postgres = cursor.fetchall()
    cursor.close()
    return data_postgres[0][0]

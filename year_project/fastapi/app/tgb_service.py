import io

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from extracting import (
    round_half_up,
    get_hog_mean,
    get_hog_std,
    get_harris_corners_count,
    get_harris_corner_mean,
    calculate_channel_average_v2,
    count_hough_circles,
)


def create_predictable_dataframe_cat_boost(image):
    with open(image, "rb") as image:
        read = image.read()
    red_channel_intensity = round_half_up(calculate_channel_average_v2(read, "R"), 5)
    blue_channel_intensity = round_half_up(calculate_channel_average_v2(read, "B"), 5)
    green_channel_intensity = round_half_up(calculate_channel_average_v2(read, "G"), 5)
    img = Image.open(io.BytesIO(read))
    hog_mean = round_half_up(get_hog_mean(img), 5)
    hog_std = round_half_up(get_hog_std(img), 5)
    hough_circle = count_hough_circles(np.array(img))
    new_row = {
        "red_channel_intensity": red_channel_intensity,
        "blue_channel_intensity": blue_channel_intensity,
        "green_channel_intensity": green_channel_intensity,
        "HOG_mean": hog_mean,
        "houghCircle": hough_circle,
        "harris_count": get_harris_corners_count(img=np.array(img)),
        "harris_count_mean": get_harris_corner_mean(img=np.array(img)),
        "HOG_std": hog_std,
    }
    data_frame = pd.DataFrame.from_dict(data=new_row, orient="index").T
    for i in range(2):
        data_frame = pd.concat([data_frame, data_frame])
    print(data_frame)
    scaler = StandardScaler()
    dataframe_train_scaled = scaler.fit_transform(data_frame)
    pca = PCA(n_components=4)
    return pca.fit_transform(dataframe_train_scaled)


def create_predictable_dataframe(image):
    with open(image, "rb") as image:
        read = image.read()
    red_channel_intensity = round_half_up(calculate_channel_average_v2(read, "R"), 5)
    blue_channel_intensity = round_half_up(calculate_channel_average_v2(read, "B"), 5)
    green_channel_intensity = round_half_up(calculate_channel_average_v2(read, "G"), 5)
    img = Image.open(io.BytesIO(read))
    hog_mean = round_half_up(get_hog_mean(img), 5)
    hog_std = round_half_up(get_hog_std(img), 5)
    new_row = {
        "red_channel_intensity": red_channel_intensity,
        "blue_channel_intensity": blue_channel_intensity,
        "green_channel_intensity": green_channel_intensity,
        "HOG_mean": hog_mean,
        "harris_count": get_harris_corners_count(img=np.array(img)),
        "harris_count_mean": get_harris_corner_mean(img=np.array(img)),
        "HOG_std": hog_std,
    }
    return pd.DataFrame.from_dict(data=new_row, orient="index").T

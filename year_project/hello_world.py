from PIL import Image, ImageChops
import os
import numpy as np
import pandas as pd
from skimage import io
from sklearn.cluster import KMeans
import random
import cv2
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from sklearn.utils import shuffle
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt

import seaborn as sns

from year_project.functions import data_dictionary, calculate_channel_average, plot_channel_intensity_barplot, \
    create_bar_plot_with_counts, set_type_array, create_concatenated_plot, create_distribution_image_size

sns.set(style="dark",
        color_codes=True,
        font_scale=1.5)

import warnings
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
np.random.seed(42)

df = data_dictionary()
create_bar_plot_with_counts(df)

red_channel_average = df.groupby("Label").agg({"image_path": lambda s: np.mean(
    [calculate_channel_average(io.imread(img), channel="R") for img in s])}).reset_index()
green_channel_average = df.groupby("Label").agg({"image_path": lambda s: np.mean(
    [calculate_channel_average(io.imread(img), channel="G") for img in s])}).reset_index()
blue_channel_average = df.groupby("Label").agg({"image_path": lambda s: np.mean(
    [calculate_channel_average(io.imread(img), channel="B") for img in s])}).reset_index()

plot_channel_intensity_barplot(red_channel_average, "r")
plot_channel_intensity_barplot(green_channel_average, "g")
plot_channel_intensity_barplot(blue_channel_average, "b")


set_type_array("red", red_channel_average)
set_type_array("green", green_channel_average)
set_type_array("blue", blue_channel_average)

channel_averages = pd.concat([red_channel_average, green_channel_average, blue_channel_average])

create_concatenated_plot(channel_averages)


df["size"] = df["image_path"].apply(np.size)

create_distribution_image_size(df)
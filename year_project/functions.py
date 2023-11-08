from skimage import io
import os
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def calculate_channel_average(img, channel):
    channel_dict = {"R": 0, "G": 1, "B": 2}
    channel_idx = channel_dict[channel]
    channel_intensities = np.array([row[:, channel_idx] for row in img]).flatten()
    return np.mean(channel_intensities)
def data_dictionary():
    path_train = "/home/roman/Documents/hse/skin_problems/train/"
    train_data_categories = os.listdir(path_train)
    list_train = train_data_categories
    train_dictionary = {"image_path": [], "target": [], "Label": [], "Image":[]}
    k = 0
    for i in list_train:
        path_train_item = path_train + i
        image_list_train = os.listdir(path_train_item)
        for j in image_list_train:
            img_path_train = path_train_item + "/" + j
            train_dictionary["image_path"].append(img_path_train)
            train_dictionary['target'].append(k)
            train_dictionary['Label'].append(i)
            train_dictionary['Image'].append(io.imread(img_path_train))
        k += 1
    return pd.DataFrame(train_dictionary)


def plot_channel_intensity_barplot(df, channel):
    title_dict = {"r": "red", "g": "green", "b": "blue"}
    palet_dict = {"r": "Reds_d", "g": "Greens_d", "b": "Blues_d"}
    plt.figure(figsize=(10, 3))

    values = df["image_path"].values
    pal = sns.color_palette(palet_dict[channel], len(values))
    rank = values.argsort().argsort()
    sns.barplot(x=df["Label"].values, y=values, palette=np.array(pal[::-1])[rank])
    plt.ylabel("Intensity")
    plt.title(f"Average {title_dict[channel]} channel intensity for classes")
    plt.xticks(rotation=90)
    plt.show()

def create_bar_plot_with_counts(df):
    labels = df.groupby("Label")["image_path"].count()
    pal = sns.color_palette("Purples_d", len(labels))
    rank = labels.argsort().argsort()
    sns.barplot(x=labels.index, y=labels.values, palette=np.array(pal[::-1])[rank])
    plt.title("Distribution of classes within dataset")
    plt.ylabel("Counts of images")
    plt.xticks(rotation=90)
    plt.show()


def set_type_array(channel,df):
    df["type"] = [channel for _ in range(len(df))]


def create_concatenated_plot(channel_averages):
    plt.figure(figsize=(15, 5))
    colors = [[0.8, 0.3, 0.3], [0.1, 0.6, 0.2], [0.2, 0.3, 0.6]]
    ax = sns.barplot(x=channel_averages["Label"].values, y="image_path", hue="type",
                     data=channel_averages, palette=colors)
    plt.xlabel("Label")
    plt.title("Average of channel intensities grouped by class")
    ax.legend(title="Channel", loc="lower right")
    plt.xticks(rotation=90)
    plt.show()

def create_distribution_image_size(df):
    plt.figure(figsize=(12, 7))
    sns.boxplot(y=df["Label"].values, x="size", data=df, orient="h")
    plt.title("Distribution of image sizes")
    plt.xticks(np.arange(0, 1900000, 200000), rotation=45)
    plt.xlim(0, 1900000)
    plt.ylabel("Label");
    plt.ylabel("Size");
    plt.show()
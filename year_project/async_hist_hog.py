import asyncio
import os
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import Pool
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from skimage import io
from skimage.feature import hog


def get_hog_features(image):
    fd, hog_image = hog(image, orientations=8,
                        pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1),
                        visualize=True,
                        channel_axis=-1)
    return fd


def get_hog_mean(img):
    print("start hog mean")
    fd = get_hog_features(img)
    return np.mean(fd) if len(fd) else 0


def get_hog_std(img):
    print("start hog std")
    fd = get_hog_features(img)
    return np.std(fd) if len(fd) else 0


async def data_dictionary(path):
    train_data_categories = os.listdir(path)
    list_train = train_data_categories
    train_dictionary = {"image_path": [], "target": [], "Label": [], "Image": []}
    k = 0
    for i in list_train:
        print(f"reading {i}")
        path_train_item = path + i
        image_list_train = os.listdir(path_train_item)
        for j in image_list_train:
            img_path_train = path_train_item + "/" + j
            train_dictionary["image_path"].append(img_path_train)
            train_dictionary['target'].append(k)
            train_dictionary['Label'].append(i)
            train_dictionary['Image'].append(await read_image(img_path_train))
        k += 1
    return pd.DataFrame(train_dictionary)


async def read_image(path):
    return io.imread(path)


async def create_hog_histogram(df):
    f, axes = plt.subplots(figsize=(40, 40), ncols=2, nrows=1)
    sns.boxplot(y=df["Label"].values, x="HOG_mean", data=df, orient="h", ax=axes[0])
    sns.boxplot(y=df["Label"].values, x="HOG_std", data=df, orient="h", ax=axes[1])
    axes[0].set_xlim(0.28, 0.36)
    axes[0].title.set_text("Distribution of HOG descriptor mean")
    axes[1].title.set_text("Distribution of HOG descriptor standard deviation")
    axes[0].set_xlabel("HOG Mean")
    axes[1].set_xlabel("HOG Standard Deviation")
    plt.show()


async def main():
    df = await data_dictionary("/home/roman/Documents/hse/skin_problems/train/")
    with Pool(12) as executor:
        df['HOG_mean'] = executor.map(get_hog_mean, df['Image'])
        df['HOG_std'] = executor.map(get_hog_std, df['Image'])

    await create_hog_histogram(df)


asyncio.run(main())

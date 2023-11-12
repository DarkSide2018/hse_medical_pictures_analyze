import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from skimage.feature import hog

from year_project.functions import data_dictionary


def get_hog_features(image):
    """
    Gets the histogram of oriented gradients for an image
    """
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                        channel_axis=-1)
    return fd

def get_hog_mean(img):
    print("work")
    fd = get_hog_features(img)
    return np.mean(fd) if len(fd) else 0


def get_hog_std(img):
    """
    Gets the standard deviation of the HOG response map
    """
    fd = get_hog_features(img)
    return np.std(fd) if len(fd) else 0


def create_hog_histogram(df):
    f, axes = plt.subplots(figsize=(40, 40), ncols=2, nrows=1)
    sns.boxplot(y=df["Label"].values, x="HOG_mean", data=df, orient="h", ax=axes[0])
    sns.boxplot(y=df["Label"].values, x="HOG_std", data=df, orient="h", ax=axes[1])
    axes[0].set_xlim(0.28, 0.36)
    axes[0].title.set_text("Distribution of HOG descriptor mean")
    axes[1].title.set_text("Distribution of HOG descriptor standard deviation")
    axes[0].set_xlabel("HOG Mean")
    axes[1].set_xlabel("HOG Standard Devioation")
    plt.show()

df = data_dictionary()

df['HOG_mean'] = df["Image"].apply(lambda img: get_hog_mean(img))
df['HOG_std'] = df["Image"].apply(lambda img: get_hog_std(img))

create_hog_histogram(df)
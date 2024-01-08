import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import roc_auc_score
import concurrent.futures
import pickle


def calculate_channel_average(img, channel):
    channel_dict = {"R": 0, "G": 1, "B": 2}
    channel_idx = channel_dict[channel]
    channel_intensities = np.array([row[:, channel_idx] for row in img]).flatten()
    return np.mean(channel_intensities)


def get_hog_features(image):
    """
    Gets the histogram of oriented gradients for an image
    """
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                        channel_axis=-1)
    return fd


def get_hog_mean(img):
    """
    Gets the mean of the HOG response map
    """
    fd = get_hog_features(img)
    return np.mean(fd) if len(fd) else 0


def get_hog_std(img):
    """
    Gets the standard deviation of the HOG response map
    """
    fd = get_hog_features(img)
    return np.std(fd) if len(fd) else 0


def process_class(j, path_train_item, k, i):
    print(f" extracting j : {j}")
    img_path_train = path_train_item + "/" + j
    image = io.imread(img_path_train)
    red_channel_intensity = calculate_channel_average(image, "R")
    blue_channel_intensity = calculate_channel_average(image, "B")
    green_channel_intensity = calculate_channel_average(image, "G")
    hog_mean = get_hog_mean(image)
    hog_std = get_hog_std(image)
    new_row = {
        'image_path': img_path_train,
        'target': k,
        'Label': i,
        'red_channel_intensity': red_channel_intensity,
        'blue_channel_intensity': blue_channel_intensity,
        'green_channel_intensity': green_channel_intensity,
        'HOG_mean': hog_mean,
        'HOG_std': hog_std,
        'Image': image
    }

    return new_row


def process_folder(train_dictionary, k, i, path_train):
    print(f" extracting i : {i}")
    path_train_item = path_train + "/" + i
    image_list_train = os.listdir(path_train_item)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        counter = 0
        for j in image_list_train:
            futures.append(
                executor.submit(process_class, j=j, path_train_item=path_train_item, k=k, i=i))
            print(f" processing {counter} in {str(len(image_list_train))}")
            counter += 1
    for future in concurrent.futures.as_completed(futures):
        train_dictionary.loc[len(train_dictionary)] = future.result()
    return train_dictionary


def data_dictionary(part="train"):
    path_train = f"/home/roman/Documents/hse/skin_problems_small/{part}/"
    train_data_categories = os.listdir(path_train)
    list_train = train_data_categories
    train_dictionary = {"image_path": [],
                        "target": [],
                        "Label": [],
                        "Image": [],
                        'HOG_mean': [],
                        'HOG_std': [],
                        'red_channel_intensity': [],
                        'blue_channel_intensity': [],
                        'green_channel_intensity': []}
    train_dictionary = pd.DataFrame(train_dictionary)
    k = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in list_train:
            futures.append(
                executor.submit(process_folder, train_dictionary=train_dictionary, k=k, i=i, path_train=path_train))
            k += 1
    for future in concurrent.futures.as_completed(futures):
        train_dictionary = pd.concat([train_dictionary, future.result()])
    return train_dictionary


dataframe_train = data_dictionary(part="train")
dataframe_test = data_dictionary(part="test")

dataframe_train.drop('image_path', axis=1, inplace=True)
dataframe_train.drop('Label', axis=1, inplace=True)
dataframe_train.drop('Image', axis=1, inplace=True)

dataframe_test.drop('image_path', axis=1, inplace=True)
dataframe_test.drop('Label', axis=1, inplace=True)
dataframe_test.drop('Image', axis=1, inplace=True)

print(dataframe_train)

y_train = dataframe_train['target']
dataframe_train.drop('target', axis=1, inplace=True)


X_train = dataframe_train


y_test = dataframe_test['target']

dataframe_test.drop('target', axis=1, inplace=True)


X_test = dataframe_test


clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

roc_auc = roc_auc_score(y_test, predicted)

print(f"roc_auc : {roc_auc}")

with open('svm_svc_rbf.pickle', 'wb') as f:
    pickle.dump(clf, f)

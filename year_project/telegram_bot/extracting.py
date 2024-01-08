import math
import numpy as np
import io
from PIL import Image
import cv2
from skimage.feature import hog


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def calculate_channel_average(img, channel):
    channel_dict = {"R": 0, "G": 1, "B": 2}
    channel_idx = channel_dict[channel]
    channel_intensities = np.array([row[:, channel_idx] for row in img]).flatten()
    return np.mean(channel_intensities)


def calculate_channel_average_v2(img_data, channel):
    img = Image.open(io.BytesIO(img_data))
    img_array = np.array(img)
    channel_dict = {"R": 0, "G": 1, "B": 2}
    channel_idx = channel_dict[channel]
    channel_intensities = img_array[:, :, channel_idx].flatten()
    return np.mean(channel_intensities)


def get_harris_corners_coordinates(img):
    """
    Returns a matrix containing the Harris Corners of an image. Coordinates mostly used for visualizing.
    """
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 200, 0.75, 100)
    return np.int0(corners) if corners is not None else np.array([])


def get_harris_corners_count(img):
    """
    Returns the number of Harris Corners in a given image
    """
    return len(get_harris_corners_coordinates(img))


def get_harris_corner_mean(img):
    """
    Returns the mean of the non-zero harris corners.
    The Harris Corners are found at the local maximas of the corners response map
    """
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    corners = np.where(corners > 0)
    return np.mean(corners)


def set_type_array(channel, df):
    df["type"] = [channel for _ in range(len(df))]


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

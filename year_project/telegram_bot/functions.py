import concurrent.futures
import os

import pandas as pd
import psycopg2
from pandas import DataFrame
from skimage import io
from sqlalchemy import create_engine

from year_project.telegram_bot.extracting import round_half_up, calculate_channel_average, get_hog_mean, get_hog_std, \
    get_harris_corners_count, get_harris_corner_mean

conn_string = 'postgresql://hse_medical:123456@localhost:5450/hse_medical'

db = create_engine(conn_string)
conn = db.connect()
conn.autocommit = True

conn_select = psycopg2.connect(
    database="hse_medical",
    user='hse_medical',
    password='123456',
    host='127.0.0.1',
    port='5450',
    options="-c search_path=analyze"
)

conn_select.autocommit = True


def get_connection():
    return conn_select


def process_class(j, path_train_item, k, i):
    cursor = conn_select.cursor()
    print(f" extracting j : {j}")
    table = path_train_item.split("/")[6]
    img_path_train = path_train_item + "/" + j
    sql1 = f'''select image_path, 
    target,
    label,
    red_channel_intensity,
    blue_channel_intensity,
    green_channel_intensity,
    hog_mean,
    harris_count,
    harris_count_mean,
    hog_std from medical_pictures_{table} where image_path = '{img_path_train}' limit 1;'''
    cursor.execute(sql1)
    try:
        data_postgres = cursor.fetchall()
        cursor.close()
        if len(data_postgres) != 0:
            frame = DataFrame(data_postgres)
            frame.columns = ['image_path',
                             'target',
                             'Label',
                             'red_channel_intensity',
                             'blue_channel_intensity',
                             'green_channel_intensity',
                             'HOG_mean',
                             'harris_count',
                             'harris_count_mean',
                             'HOG_std']
            return frame
    except Exception as error:
        print("An exception occurred during process_class:", error)
    finally:
        cursor.close()
    print(f" reading from file system j : {j}")
    image = io.imread(img_path_train)
    red_channel_intensity = round_half_up(calculate_channel_average(image, "R"), 5)
    blue_channel_intensity = round_half_up(calculate_channel_average(image, "B"), 5)
    green_channel_intensity = round_half_up(calculate_channel_average(image, "G"), 5)
    hog_mean = round_half_up(get_hog_mean(image), 5)
    hog_std = round_half_up(get_hog_std(image), 5)
    sql_get_target = f'''select target from target_dictionary where label = '{i}';'''
    target_cursor = conn_select.cursor()
    target_cursor.execute(sql_get_target)
    target = target_cursor.fetchall()
    new_row = {
        'image_path': img_path_train,
        'target': target[0][0],
        'Label': i,
        'red_channel_intensity': red_channel_intensity,
        'blue_channel_intensity': blue_channel_intensity,
        'green_channel_intensity': green_channel_intensity,
        'HOG_mean': hog_mean,
        'harris_count': get_harris_corners_count(image),
        'harris_count_mean': get_harris_corner_mean(image),
        'HOG_std': hog_std
    }
    target_cursor.close()
    table_name = f"medical_pictures_{table}"
    keys = new_row.keys()
    columns = ','.join(keys)
    insert = 'insert into {1} ({0}) values (%s, %s, %s, %s,%s,%s,%s,%s)'.format(columns, table_name)
    insert_cursor = conn_select.cursor()
    try:
        insert_cursor.execute(insert,
                              (
                                  new_row['image_path'],
                                  new_row['target'],
                                  new_row['Label'],
                                  new_row['red_channel_intensity'],
                                  new_row['blue_channel_intensity'],
                                  new_row['green_channel_intensity'],
                                  new_row['HOG_mean'],
                                  new_row['harris_count'],
                                  new_row['harris_count_mean'],
                                  new_row['HOG_std'])
                              )
        insert_cursor.close()
    except Exception as error:
        print("An exception occurred process_class after insert:", error)
    print(f"new_row {new_row}")
    return pd.DataFrame.from_dict(data=new_row, orient='index')


def process_folder(train_dictionary, k, i, path_train):
    print(f" extracting i : {i}")
    path_train_item = path_train + i
    print(path_train_item)
    image_list_train = os.listdir(path_train_item)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for j in image_list_train:
            futures.append(
                executor.submit(process_class, j=j, path_train_item=path_train_item, k=k, i=i))
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(f"describe: {result.describe()} {len(result.columns)} {result.columns}")
        print(f"describe train dictionary: {train_dictionary.describe()} {train_dictionary.columns}")
        train_dictionary = pd.concat([train_dictionary, result], ignore_index=True)
    return train_dictionary


def data_dictionary(part="train"):
    path_train = f"/home/roman/Documents/hse/skin_problems_small/{part}/"
    train_data_categories = os.listdir(path_train)
    list_train = train_data_categories
    train_dictionary = {'image_path': [],
                        'target': [],
                        'Label': [],
                        'red_channel_intensity': [],
                        'blue_channel_intensity': [],
                        'green_channel_intensity': [],
                        'HOG_mean': [],
                        'harris_count': [],
                        'harris_count_mean': [],
                        'HOG_std': []}
    train_dictionary = pd.DataFrame(train_dictionary)
    print(train_dictionary)
    k = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in list_train:
            futures.append(
                executor.submit(process_folder, train_dictionary=train_dictionary, k=k, i=i, path_train=path_train))
            k += 1
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(f"describe: {result.describe()} {len(result.columns)}")
        train_dictionary = pd.concat([train_dictionary, result])
    train_dictionary.drop('image_path', axis=1, inplace=True)
    train_dictionary.drop('Label', axis=1, inplace=True)
    train_dictionary.columns = train_dictionary.columns.astype(str)
    # train_dictionary.drop('0', axis=1, inplace=True)
    print(f"nulls in dataframe train_dictionary : {train_dictionary.isnull().sum()}")
    train_dictionary = train_dictionary.dropna()
    return train_dictionary

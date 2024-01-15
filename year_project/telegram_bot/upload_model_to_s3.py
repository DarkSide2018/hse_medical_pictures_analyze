import os

import boto3
from dotenv import load_dotenv

load_dotenv()

access_key = os.environ.get('aws_access_key_id')
secret_key = os.environ.get('aws_secret_access_key')
region = 'ru-central1-a'

# Инициализация клиента S3
s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)

# Загрузка файла в бакет
bucket_name = 'cat-boost-model'
file_path = 'models/gs_cb_best_estimator_.pickle'
object_name = 'gs_cb_best_estimator'


def upload_file_to_bucket(value):
    s3.upload_file(file_path, bucket_name, object_name)


def model_from_s3(model_name):
    s3.download_file(bucket_name, model_name, f"{model_name}.pickle")

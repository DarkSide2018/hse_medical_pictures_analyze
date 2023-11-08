import os

import boto3


def uploadDirectory(path, bucket):
    for root, dirs, files in os.walk(path):
        for file in files:
            bucket.upload_file(os.path.join(root, file), file)

def get_bucket_by_name(name):
    s3 = boto3.resource(
    service_name='s3',
    region_name='ru-central-1',
    aws_access_key_id=os.environ.get('aws_access_key_id'),
    aws_secret_access_key=os.environ.get('aws_secret_access_key')
    )
    for bucket in s3.buckets.all():
        if(bucket.name == name):
            return bucket


# bucket = get_bucket_by_name("hse-medical-pictures")

s3client = boto3.client('s3')
bucket = 'hse-medical-pictures'
startAfter = 'train/'

theobjects = s3client.list_objects_v2(Bucket=bucket, StartAfter=startAfter)

for object in theobjects['Contents']:
    print (object['Key'])

# for obj in bucket.objects.all():
#     key = obj.key
#     print(f"key - {key}")
#     body = obj.get()['Body'].read()
#     print(f"body - {body}")



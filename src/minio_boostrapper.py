import os
import logging
from minio import Minio
from dotenv import load_dotenv
from utils import setup_logger


class MinioClientBootstrapper:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        self.client = Minio(endpoint, access_key, secret_key, secure=secure)

    def create_bucket_if_not_exists(self, bucket_name):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            logging.info(f"Bucket '{bucket_name}' has been created.")
        else:
            logging.info(f"Bucket '{bucket_name}' already exists.")

    def bootstrap_buckets(self):
        self.create_bucket_if_not_exists("mlflow")
        self.create_bucket_if_not_exists("data")


if __name__ == '__main__':
    load_dotenv()
    logging = setup_logger()

    ENDPOINT = os.getenv('MINIO_ENDPOINT')
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    SECURE = os.getenv('MINIO_SECURE').lower() == True

    minio_bootstrapper = MinioClientBootstrapper(
        endpoint=ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=SECURE
    )

    minio_bootstrapper.bootstrap_buckets()

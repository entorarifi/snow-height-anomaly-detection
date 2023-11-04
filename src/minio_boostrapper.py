import os
import logging
from minio import Minio
from dotenv import load_dotenv
from utils import setup_logger


class MinioClientBootstrapper:
    def __init__(self, endpoint, port, access_key, secret_key, secure, mlflow_bucket):
        self.client = Minio(f"{endpoint}:{port}", access_key, secret_key, secure=secure)
        self.mlflow_bucket = mlflow_bucket

    def create_bucket_if_not_exists(self, bucket_name):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            logging.info(f"Bucket '{bucket_name}' has been created")
        else:
            logging.info(f"Bucket '{bucket_name}' already exists")

    def bootstrap_buckets(self):
        self.create_bucket_if_not_exists(self.mlflow_bucket)
        self.create_bucket_if_not_exists("data")


if __name__ == '__main__':
    load_dotenv()
    logging = setup_logger()

    URL = os.getenv('MINIO_URL')
    PORT = os.getenv('MINIO_PORT')
    ACCESS_KEY = os.getenv('MINIO_ROOT_USER')
    SECRET_KEY = os.getenv('MINIO_ROOT_PASSWORD')
    SECURE = os.getenv('MINIO_STORAGE_USE_HTTPS').lower() == True
    MLFLOW_BUCKET_NAME = os.getenv('MLFLOW_BUCKET_NAME')

    minio_bootstrapper = MinioClientBootstrapper(
        endpoint=URL,
        port=PORT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=SECURE,
        mlflow_bucket=MLFLOW_BUCKET_NAME
    )

    minio_bootstrapper.bootstrap_buckets()

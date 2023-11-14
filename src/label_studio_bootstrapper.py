import logging
import os
from io import BytesIO

import pandas as pd
import requests
from dotenv import load_dotenv
from label_studio_sdk import Project
from label_studio_sdk.project import ProjectStorage
from minio import Minio
from minio.deleteobjects import DeleteObject

from label_studio_client import LabelStudioClient
from utils import setup_logger


# noinspection PyTypeChecker
class LabelStudioClientBootstrapper(LabelStudioClient):
    def __init__(
            self,
            base_url,
            api_key,
            project_name,
            labeled_train_data_path,
            label_config_path,
            s3_url,
            s3_user,
            s3_password,
            s3_secure,
            s3_bucket_name
    ):
        super().__init__(base_url, api_key, project_name)
        self.labeled_train_data_path = labeled_train_data_path
        self.label_config_path = label_config_path
        self.s3_url = s3_url
        self.s3_user = s3_user
        self.s3_password = s3_password
        self.s3_secure = s3_secure
        self.s3_bucket_name = s3_bucket_name

        self.minio_client = Minio(self.s3_url, self.s3_user, self.s3_password, secure=self.s3_secure)

    def create_project_if_not_exists(self):
        if self.get_project()[0] is None:
            with open(self.label_config_path) as xml:
                project: Project = self.client.create_project(
                    title=self.project_name,
                    label_config=xml.read()
                )

                logging.info("Project has been created")

                return project

    def get_storage(self):
        project_id = self.project_id
        url = f"{self.base_url}/api/storages"

        response = requests.get(url=url, params={'project': project_id}, headers=self.headers)

        if response.status_code == 200:
            for storage in response.json():
                if storage['title'] == self.s3_bucket_name:
                    return storage

            return None

        raise Exception(f'Failed to retrieve data: {response.status_code}')

    def connect_to_s3_storage(self):
        storage = self.get_storage()

        if storage is None:
            logging.info("Connecting to s3 storage...")
            storage = self.project.connect_s3_import_storage(
                bucket=self.s3_bucket_name,
                title=self.s3_bucket_name,
                s3_endpoint="http://minio:9000",  # TODO: Replace with env
                aws_access_key_id=self.s3_user,
                aws_secret_access_key=self.s3_password,
                presign_ttl=60,
                use_blob_urls=False
            )
            logging.info("Successfully connected to storage")

            self.project.sync_storage(ProjectStorage.S3.value, storage['id'])
        else:
            logging.info("Storage already exists")

    def bootstrap_project(self):
        self.project = self.create_project_if_not_exists()
        self.connect_to_s3_storage()
        tasks = self.upload_data_to_s3_and_create_tasks()
        logging.info('Importing tasks...')
        ids = self.project.import_tasks(tasks)
        logging.info(f"{len(ids)} tasks have been imported")

    def upload_data_to_s3_and_create_tasks(self):
        objects_to_delete = [
            DeleteObject(obj.object_name)
            for obj in self.minio_client.list_objects(self.s3_bucket_name, recursive=True)
        ]

        self.minio_client.remove_objects(self.s3_bucket_name, objects_to_delete)

        files = os.listdir(self.labeled_train_data_path)

        tasks = []

        for f in files:
            station_df = pd.read_csv(os.path.join(self.labeled_train_data_path, f), index_col=False)
            station_df = station_df[['station_code', 'measure_date', 'HS']]
            station_code = station_df.iloc[0]['station_code']
            file_name = f"{station_code}.csv"
            station_df['measure_date'] = pd.to_datetime(station_df['measure_date']).dt.tz_localize(None)

            csv_bytes = station_df.to_csv(index=False).encode('UTF-8')
            buffer = BytesIO(csv_bytes)

            self.minio_client.put_object(self.s3_bucket_name, file_name, buffer, len(csv_bytes))

            json_task = {
                "data": {
                    "csv": f"s3://data/{station_code}.csv",
                    "start_date": station_df.iloc[0]['measure_date'].strftime('%Y-%m-%d'),
                    "end_date": station_df.iloc[-1]['measure_date'].strftime('%Y-%m-%d'),
                    "number_of_datapoints": len(station_df),
                    "station_code": station_code
                }
            }

            tasks.append(json_task)

        return tasks


if __name__ == '__main__':
    load_dotenv()
    logging = setup_logger()

    BASE_URL = os.getenv('LS_BASE_URL')
    API_KEY = os.getenv('LS_API_KEY')
    PROJECT_NAME = os.getenv('LS_PROJECT_NAME')
    LABELED_TRAIN_DATA_PATH = os.getenv('LS_LABELED_TRAIN_DATA_PATH')
    LABEL_CONFIG_PATH = os.getenv('LS_LABEL_CONFIG_PATH')

    S3_USER = os.getenv('MINIO_ROOT_USER')
    S3_PASSWORD = os.getenv('MINIO_ROOT_PASSWORD')
    S3_URL = os.getenv('MINIO_URL')
    S3_PORT = os.getenv('MINIO_PORT')
    S3_SECURE = os.getenv('MINIO_STORAGE_USE_HTTPS').lower() is True
    S3_BUCKET_NAME = os.getenv('LS_BUCKET_NAME')

    ls = LabelStudioClientBootstrapper(
        base_url=BASE_URL,
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        labeled_train_data_path=LABELED_TRAIN_DATA_PATH,
        label_config_path=LABEL_CONFIG_PATH,
        s3_url=f'{S3_URL}:{S3_PORT}',
        s3_user=S3_USER,
        s3_password=S3_PASSWORD,
        s3_secure=S3_SECURE,
        s3_bucket_name=S3_BUCKET_NAME
    )

    logging.info(f'Deleting all projects...')
    responses = ls.client.delete_all_projects()
    logging.info(f'{len(responses)} project(s) have been deleted')
    ls.bootstrap_project()

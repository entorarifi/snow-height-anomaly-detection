import logging
import os
import json
import requests
import pandas as pd

from label_studio_sdk import Client, Project
from dotenv import load_dotenv
from utils import setup_logger


class LabelStudioBootstrapper:
    def __init__(
            self,
            base_url,
            api_key,
            project_name,
            csv_path,
            split_csv_path,
            label_config_path,
            local_storage_title,
            local_storage_path,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.project_name = project_name
        self.csv_path = csv_path
        self.split_csv_path = split_csv_path
        self.label_config_path = label_config_path
        self.local_storage_title = local_storage_title
        self.local_storage_path = local_storage_path
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.client: Client = self.get_client()
        self.project: Project = self.get_project()

    def get_client(self):
        return Client(url=self.base_url, api_key=self.api_key)

    def get_project(self):
        projects = self.client.get_projects()

        for p in projects:
            if p.get_params().get("title") == self.project_name:
                return p

        return None

    def get_local_storage(self):
        project_id = self.project.get_params().get('id')
        url = f"{self.base_url}/api/storages"

        response = requests.get(url=url, params={'project': project_id}, headers=self.headers)

        if response.status_code == 200:
            for storage in response.json():
                if storage['title'] == self.local_storage_title:
                    return storage

            return None

        raise Exception(f'Failed to retrieve data: {response.status_code}')

    def create_storage_if_not_exists(self):
        local_storage = self.get_local_storage()

        storage_id = None

        if local_storage is None:
            logging.info("Creating storage...")
            storage_id = self.create_local_storage()
        else:
            logging.info("Storage already exists")

        return storage_id if local_storage is None else local_storage['id']

    def create_local_storage(self):
        project_id = self.project.get_params().get('id')
        url = f"{self.base_url}/api/storages/localfiles"

        data = {
            "path": self.local_storage_path,
            "title": self.local_storage_title,
            "project": project_id,
            "use_blob_urls": True
        }

        response = requests.post(url=url, headers=self.headers, data=json.dumps(data))

        if response.status_code == 201:
            return response.json()['id']

        raise Exception(f'Failed to retrieve data: {response.status_code}')

    def bootstrap_project(self):
        ls.split_data_into_stations()

        if self.project is None:
            with open(LABEL_CONFIG_PATH) as xml:
                self.project = ls.client.create_project(
                    title=self.project_name,
                    label_config=xml.read()
                )
            logging.info("Project has been created")

        storage_id = self.create_storage_if_not_exists()

        self.project.sync_storage(
            storage_type='localfiles',
            storage_id=storage_id
        )
        logging.info("Storage has been synced")

    def split_data_into_stations(self, overwrite=False):
        df = pd.read_csv(self.csv_path)

        if os.path.isdir(self.split_csv_path):
            logging.info("Data has already been split")

            if not overwrite:
                return

            logging.info("Overwriting...")

        else:
            os.mkdir(self.split_csv_path)
            logging.info("Splitting data...")

        def station_to_csv(x):
            station_code = x.iloc[0]['station_code']
            file_name = f"{self.split_csv_path}/{station_code}.csv"
            x['measure_date'] = pd.to_datetime(x['measure_date']).dt.tz_localize(None)
            x.to_csv(file_name, index=False)

        df.groupby('station_code').apply(lambda x: station_to_csv(x))


if __name__ == '__main__':
    load_dotenv()
    logging = setup_logger()

    BASE_URL = os.getenv('LS_BASE_URL')
    API_KEY = os.getenv('LS_API_KEY')
    PROJECT_NAME = os.getenv('LS_PROJECT_NAME')
    ORIGINAL_CSV_FILE_PATH = os.getenv('LS_ORIGINAL_CSV_FILE_PATH')
    SPLIT_CSV_DIR_PATH = os.getenv('LS_SPLIT_CSV_DIR_PATH')
    LABEL_CONFIG_PATH = os.getenv('LS_LABEL_CONFIG_PATH')
    LOCAL_STORAGE_TITLE = os.getenv('LS_LOCAL_STORAGE_TITLE')
    LOCAL_STORAGE_PATH = os.getenv('LS_LOCAL_STORAGE_PATH')

    ls = LabelStudioBootstrapper(
        base_url=BASE_URL,
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        csv_path=ORIGINAL_CSV_FILE_PATH,
        split_csv_path=SPLIT_CSV_DIR_PATH,
        label_config_path=LABEL_CONFIG_PATH,
        local_storage_title=LOCAL_STORAGE_TITLE,
        local_storage_path=LOCAL_STORAGE_PATH
    )

    ls.bootstrap_project()

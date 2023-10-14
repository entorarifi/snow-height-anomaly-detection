import logging
import os

import pandas as pd

from label_studio_sdk import Client, Project
from dotenv import load_dotenv
from utils import setup_logger


class LabelStudioBootstrapper:
    def __init__(self, url, api_key, csv_path, split_csv_path):
        self.url = url
        self.api_key = api_key
        # self.client: Client = self.create_client()
        # self.project: Project = self.get_project()
        self.csv_path = csv_path
        self.split_csv_path = split_csv_path

    def create_client(self):
        return Client(url=self.url, api_key=self.api_key)

    def get_project(self):
        projects = self.client.get_projects()

        if len(projects) == 0:
            logging.error("Couldn't find any projects.")
            return

        return projects[0]

    def split_data_into_stations(self):
        df = pd.read_csv(self.csv_path)

        if os.path.isdir('%s' % self.split_csv_path):
            logging.info("Data has already been split")
            return
        else:
            os.mkdir("%s" % self.split_csv_path)

        logging.info("Splitting data...")

        def get_file_name(x):
            return "%s/%s.csv" % (self.split_csv_path, x.iloc[0]['station_code'])

        df.groupby('station_code').apply(lambda x: x.to_csv(get_file_name(x), index=False))


if __name__ == '__main__':
    load_dotenv()
    logging = setup_logger()

    API_URL = os.getenv('LS_API_URL')
    API_KEY = os.getenv('LS_API_KEY')
    ORIGINAL_CSV_FILE = os.getenv('LS_ORIGINAL_CSV_FILE')
    SPLIT_CSV_DIR = os.getenv('LS_SPLIT_CSV_DIR')

    ls = LabelStudioBootstrapper(API_URL, API_KEY, ORIGINAL_CSV_FILE, SPLIT_CSV_DIR)

    ls.split_data_into_stations()

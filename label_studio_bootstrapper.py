import logging
import os

import pandas as pd
from dotenv import load_dotenv
from label_studio_sdk import Project

from label_studio_client import LabelStudioClient
from utils import setup_logger


class LabelStudioClientBootstrapper(LabelStudioClient):
    def __init__(self, base_url, api_key, project_name, daily_snow_csv_path, stations_csv_path, label_config_path):
        super().__init__(base_url, api_key, project_name)
        self.daily_snow_csv_path = daily_snow_csv_path
        self.stations_csv_path = stations_csv_path
        self.label_config_path = label_config_path
        self.tasks = []

    def create_project_if_not_exists(self):
        if self.get_project() is None:
            with open(LABEL_CONFIG_PATH) as xml:
                project: Project = ls.client.create_project(
                    title=self.project_name,
                    label_config=xml.read()
                )

                logging.info("Project has been created")

                return project

    def bootstrap_project(self):
        ls.group_data_by_station()

        self.project = self.create_project_if_not_exists()

        logging.info('Importing tasks...')
        ids = self.project.import_tasks(self.tasks)
        logging.info(f"{len(ids)} tasks were imported")

    def group_data_by_station(self):
        df = pd.read_csv(self.daily_snow_csv_path)

        if os.path.isdir(self.stations_csv_path):
            logging.info("Stations dir exists. Overwriting...")

        else:
            os.mkdir(self.stations_csv_path)
            logging.info("Grouping data into separate CSVs...")

        def station_to_csv(x):
            station_code = x.iloc[0]['station_code']
            file_name = f"{self.stations_csv_path}/{station_code}.csv"
            x['measure_date'] = pd.to_datetime(x['measure_date']).dt.tz_localize(None)
            x.to_csv(file_name, index=False)

            json_task = {
                "data": {
                    "csv": f"http://localhost:8888/stations/{station_code}.csv",
                    "start_date": x.iloc[0]['measure_date'].strftime('%Y-%m-%d'),
                    "end_date": x.iloc[-1]['measure_date'].strftime('%Y-%m-%d'),
                    "number_of_datapoints": len(x),
                    "station_code": station_code
                }
            }

            self.tasks.append(json_task)

        df.groupby('station_code').apply(lambda x: station_to_csv(x))


if __name__ == '__main__':
    load_dotenv()
    logging = setup_logger()

    BASE_URL = os.getenv('LS_BASE_URL')
    API_KEY = os.getenv('LS_API_KEY')
    PROJECT_NAME = os.getenv('LS_PROJECT_NAME')
    DAILY_SNOW_CSV_PATH = os.getenv('LS_DAILY_SNOW_CSV_PATH')
    STATIONS_CSV_PATH = os.getenv('LS_STATIONS_CSV_PATH')
    LABEL_CONFIG_PATH = os.getenv('LS_LABEL_CONFIG_PATH')

    ls = LabelStudioClientBootstrapper(
        base_url=BASE_URL,
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        daily_snow_csv_path=DAILY_SNOW_CSV_PATH,
        stations_csv_path=STATIONS_CSV_PATH,
        label_config_path=LABEL_CONFIG_PATH,
    )

    ls.client.delete_all_projects()
    ls.bootstrap_project()

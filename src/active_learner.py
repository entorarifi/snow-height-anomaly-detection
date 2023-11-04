import os
import random
import logging
from io import BytesIO

import keras
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

from label_studio_client import LabelStudioClient
from utils import setup_logger


class MonteCarloDropout(keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


# noinspection PyDefaultArgument
class ActiveLearner(LabelStudioClient):
    SEQUENCE_LENGTH = 30
    TARGET_START_INDEX = SEQUENCE_LENGTH - 1
    FEATURE_COLUMNS = ['HS', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    TARGET_COLUMN = 'no_snow'
    DATE_COLUMN = 'measure_date'
    SPLIT_PERCENTAGE = 0.8
    DATASET_BATCH_SIZE = 64
    UNCERTAINTY_ITERATIONS = 5

    def __init__(self, base_url, api_key, project_name):
        super().__init__(base_url, api_key, project_name)

    def preprocces(self, df):
        df[self.DATE_COLUMN] = pd.to_datetime(df[self.DATE_COLUMN])
        df['year'] = df[self.DATE_COLUMN].dt.year
        df['month'] = df[self.DATE_COLUMN].dt.month
        df['day'] = df[self.DATE_COLUMN].dt.day
        df['weekday'] = df[self.DATE_COLUMN].dt.weekday
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        return df

    @staticmethod
    def get_features_and_targets(
            df,
            split_percentage=SPLIT_PERCENTAGE,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            scale=True,
            mean=None,
            std=None
    ):
        if (mean is None) ^ (std is None):
            raise Exception('mean and std must both be set or unset')

        features = df[feature_columns].values
        targets = None if target_column is None else df[target_column].values

        if not scale:
            return features, targets, None, None

        if mean is None:
            num_train_samples = int(len(df) * split_percentage)
            mean = features[:num_train_samples].mean(axis=0)
            features -= mean
            std = features[:num_train_samples].std(axis=0)
            features /= std
        else:
            features = (features - mean) / std

        return features, targets, mean, std

    def create_dataset(self, features, targets, shuffle=False, start_index=None, end_index=None):
        return keras.utils.timeseries_dataset_from_array(
            data=features,
            targets=targets[self.TARGET_START_INDEX:],
            sequence_length=self.SEQUENCE_LENGTH,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.DATASET_BATCH_SIZE,
            shuffle=shuffle,
            start_index=start_index,
            end_index=end_index
        )

    @staticmethod
    def predict_with_uncertainty(model, x, n_iter=UNCERTAINTY_ITERATIONS):
        predictions = np.array([model.predict(x, verbose=0) for _ in range(n_iter)])
        uncertainty = np.std(predictions, axis=0)
        scaled_uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        return predictions.mean(axis=0).reshape(-1), scaled_uncertainty.mean()

    def generate_payload(self, df):
        # Mark the start and end of each range
        df['start_range'] = df[self.TARGET_COLUMN] & (~df[self.TARGET_COLUMN].shift(1, fill_value=False))
        df['end_range'] = df[self.TARGET_COLUMN] & (~df[self.TARGET_COLUMN].shift(-1, fill_value=False))

        # Get start dates and end dates
        start_dates = df[df['start_range']][self.DATE_COLUMN]
        end_dates = df[df['end_range']][self.DATE_COLUMN].reset_index(drop=True)

        # Generate payload for dataset
        payload = [
            {
                "type": "timeserieslabels",
                "value": {
                    "start": start.strftime('%Y-%m-%d'),
                    "end": end.strftime('%Y-%m-%d'),
                    "instant": start == end,
                    "timeserieslabels": ["Outlier"]
                },
                "to_name": "ts",
                "from_name": "label"
            }
            for start, end in zip(start_dates, end_dates)
        ]

        return payload

    def get_most_uncertain_prediction(self):
        unlabeled_tasks = self.project.get_unlabeled_tasks()

        most_uncertain = {
            'predictions_score': 0
        }

        for task in unlabeled_tasks:
            if task['predictions_score'] > most_uncertain['predictions_score']:
                most_uncertain = task

        return most_uncertain

    def parse_df(self, task):
        # TODO: Test whether dfs are being parsed accordingly
        path = task['data']['csv']
        labels = task['annotations'][-1]['result']

        start_dates = []
        end_dates = []

        for label in labels:
            start_dates.append(pd.Timestamp(label['value']['start']))
            end_dates.append(pd.Timestamp(label['value']['end']))

        response = requests.get(url=f"http://localhost:8080{path}", headers=self.headers)

        if response.status_code == 200:
            df = pd.read_csv(BytesIO(response.content), encoding='UTF-8').dropna()
        else:
            raise Exception(f'Failed to retrieve data: {response.status_code}')

        df[self.DATE_COLUMN] = pd.to_datetime(df[self.DATE_COLUMN])

        def is_outlier(date):
            for start, end in zip(start_dates, end_dates):
                if start <= date <= end:
                    return 1
            return 0

        df[self.TARGET_COLUMN] = df[self.DATE_COLUMN].apply(is_outlier)

        return df

    def create_train_val_datasets(self, tasks):
        train_df, val_df = pd.DataFrame(), pd.DataFrame()

        for task in tasks:
            df = self.parse_df(task)

            split_index = int(len(df) * self.SPLIT_PERCENTAGE)

            train_df = pd.concat([train_df, df[:split_index]])
            val_df = pd.concat([val_df, df[split_index:]])

        combined_df = self.preprocces(pd.concat([train_df, val_df]))
        split_index = len(train_df)

        logging.info(f'Training samples: {len(train_df)}')
        logging.info(f'Validation samples: {len(val_df)}')

        features, targets, mean, std = self.get_features_and_targets(combined_df, split_index)

        train_dataset = self.create_dataset(features, targets, end_index=split_index, shuffle=True)
        val_dataset = self.create_dataset(features, targets, start_index=split_index, shuffle=True)

        return train_dataset, val_dataset

    def run_iteration(self):
        # 1. Pick random station
        unlabeled_tasks = self.project.get_unlabeled_tasks()
        # task = random.choice(unlabeled_tasks)
        task = unlabeled_tasks[0]

        # 2. Ask for labels.
        # TODO: This could potentially be a chrome notification or an email to annotators to label a specific station.

        # 3. Label data
        # TODO: Should be extracted into a util function and not be a part of the active learning loop because
        #  it is considered that a human expert will label the data
        task_id = task['id']
        station_code = task['data']['station_code']
        labeled_df = pd.read_csv(f'../data/labeled_daily/{station_code}.csv', index_col=False)
        # TODO: Remove this in favor of a unified generate_payload method
        labeled_df[self.DATE_COLUMN] = pd.to_datetime(labeled_df[self.DATE_COLUMN])
        payload = self.generate_payload(labeled_df)
        self.project.create_annotation(task_id, result=payload)

        # 4. Parse and split labeled data
        labeled_tasks = self.project.get_labeled_tasks()
        train_dataset, val_dataset = self.create_train_val_datasets(labeled_tasks)

        # 5. Fit model

        # 5. Assign uncertainty scores

    def predict(self):
        # tasks = self.project.get_paginated_tasks(page_size=10, page=1)['tasks']
        tasks = self.project.get_unlabeled_tasks()
        data = {}

        for task in tasks:
            # TODO: Replace url with env variable
            response = requests.get(url=f"http://localhost:8080{task['data']['csv']}", headers=self.headers)

            if response.status_code == 200:
                data[task['data']['station_code']] = {
                    'task_id': task['id'],
                    'csv': task['data']['csv'],
                    'df': pd.read_csv(BytesIO(response.content), encoding='UTF-8').dropna()
                }
            else:
                raise Exception(f'Failed to retrieve data: {response.status_code}')

        # TODO: Predict with newly created model
        model = keras.models.load_model(
            '../active-learning-results/2023-11-02/model.keras',
            custom_objects={'MonteCarloDropout': MonteCarloDropout}
        )

        for k in data.keys():
            # Preprocess
            df = self.preprocces(data[k]['df'])

            # Get features and targets
            features, targets, _, _ = self.get_features_and_targets(df, target_column=None, split_percentage=1)

            # Create dataset
            dataset = self.create_dataset(features, targets)

            # Predict
            predictions, uncertainty_score = self.predict_with_uncertainty(model, dataset)

            # Process predictions
            df[self.TARGET_COLUMN] = False
            df.loc[self.TARGET_START_INDEX:, self.TARGET_COLUMN] = predictions > 0.5

            payload = self.generate_payload(df)

            self.project.create_prediction(task_id=data[k]['task_id'], result=payload, score=float(uncertainty_score))

    def display_time_series(self, task):
        df = self.parse_df(task)

        fig = px.line(df, y="HS")

        # Create line plot using plotly express
        fig = px.line(df, y="HS", title="Time Series with Outliers")

        # Add outlier trace
        fig.add_trace(go.Scatter(
            x=df[df['outlier'] == 1][self.DATE_COLUMN],
            y=df[df['outlier'] == 1]['HS'],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=8)
        ))
        fig.show()

    def purge_annotations(self):
        url = f'http://localhost:8080/api/dm/actions?id=delete_tasks_annotations&project={self.project_id}'

        response = requests.post(url=url, headers=self.headers)

        if response.status_code == 200:
            json_response = response.json()
            logging.info(f"All ({json_response['processed_items']}) annotations were deleted")
        else:
            raise Exception(f'Failed to retrieve data: {response.status_code}')


if __name__ == '__main__':
    load_dotenv()
    logging = setup_logger(level=logging.DEBUG)

    BASE_URL = os.getenv('LS_BASE_URL')
    API_KEY = os.getenv('LS_API_KEY')
    PROJECT_NAME = os.getenv('LS_PROJECT_NAME')

    active_learner = ActiveLearner(
        base_url=BASE_URL,
        api_key=API_KEY,
        project_name=PROJECT_NAME,
    )

    # Set initial predictions
    active_learner.purge_annotations()
    active_learner.run_iteration()

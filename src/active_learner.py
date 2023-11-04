import os
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
    TARGET_COLUMNS = ['no_snow']
    SPLIT_PERCENTAGE = 0.8
    DATASET_BATCH_SIZE = 64
    UNCERTAINTY_ITERATIONS = 5

    def __init__(self, base_url, api_key, project_name):
        super().__init__(base_url, api_key, project_name)

    @staticmethod
    def preprocces(df):
        df['measure_date'] = pd.to_datetime(df['measure_date'])
        df['year'] = df['measure_date'].dt.year
        df['month'] = df['measure_date'].dt.month
        df['day'] = df['measure_date'].dt.day
        df['weekday'] = df['measure_date'].dt.weekday
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
        target_columns=TARGET_COLUMNS,
        scale=True,
        mean=None,
        std=None
    ):
        if (mean is None) ^ (std is None):
            raise Exception('mean and std must both be set or unset')

        features = df[feature_columns].values
        targets = df[target_columns].values if len(target_columns) == 0 else None

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

    def predict(self):
        # tasks = self.project.get_paginated_tasks(page_size=10, page=1)['tasks']
        tasks = self.project.get_unlabeled_tasks()
        data = {}

        for task in tasks:
            response = requests.get(url=f"http://localhost:8080{task['data']['csv']}", headers=self.headers)

            if response.status_code == 200:
                data[task['data']['station_code']] = {
                    'task_id': task['id'],
                    'csv': task['data']['csv'],
                    'df': pd.read_csv(BytesIO(response.content), encoding='UTF-8').dropna()
                }
            else:
                raise Exception(f'Failed to retrieve data: {response.status_code}')

        # Load Model
        model = keras.models.load_model(
            '../active-learning-results/2023-11-02/model.keras',
            custom_objects={'MonteCarloDropout': MonteCarloDropout}
        )

        for k in data.keys():
            # Preprocess
            df = self.preprocces(data[k]['df'])

            # Get features and targets
            features, targets, _, _ = self.get_features_and_targets(df, target_columns=[], split_percentage=1)

            # Create dataset
            dataset = self.create_dataset(features, targets)

            # Predict
            predictions, uncertainty_score = self.predict_with_uncertainty(model, dataset)

            # Process predictions
            df['no_snow'] = False
            df.loc[self.TARGET_START_INDEX:, 'no_snow'] = predictions > 0.5

            # Mark the start and end of each range
            df['start_range'] = df['no_snow'] & (~df['no_snow'].shift(1, fill_value=False))
            df['end_range'] = df['no_snow'] & (~df['no_snow'].shift(-1, fill_value=False))

            # Get start dates and end dates
            start_dates = df[df['start_range']]['measure_date']
            end_dates = df[df['end_range']]['measure_date'].reset_index(drop=True)

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

            self.project.create_prediction(task_id=data[k]['task_id'], result=payload, score=float(uncertainty_score))

    def display_time_series(self):
        tasks = self.project.get_labeled_tasks()

        path = tasks[0]['data']['csv']
        labels = tasks[0]['annotations'][0]['result']

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

        df['measure_date'] = pd.to_datetime(df['measure_date'])
        df.set_index('measure_date', inplace=True)  # set measure_date as the index

        def is_outlier(date):
            for start, end in zip(start_dates, end_dates):
                if start <= date <= end:
                    return 1
            return 0

        df['outlier'] = df.index.to_series().apply(is_outlier)

        fig = px.line(df, y="HS")

        # Create line plot using plotly express
        fig = px.line(df, y="HS", title="Time Series with Outliers")

        # Add outlier trace
        fig.add_trace(go.Scatter(
            x=df[df['outlier'] == 1].index,
            y=df[df['outlier'] == 1]['HS'],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=8)
        ))
        fig.show()


if __name__ == '__main__':
    load_dotenv()
    logging = setup_logger()

    BASE_URL = os.getenv('LS_BASE_URL')
    API_KEY = os.getenv('LS_API_KEY')
    PROJECT_NAME = os.getenv('LS_PROJECT_NAME')

    active_learner = ActiveLearner(
        base_url=BASE_URL,
        api_key=API_KEY,
        project_name=PROJECT_NAME,
    )

    # Set initial predictions
    active_learner.predict()

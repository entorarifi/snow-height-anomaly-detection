import logging
import os
import re
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
    # Dataset preparation and feature engineering
    SEQUENCE_LENGTH = 30
    TARGET_START_INDEX = SEQUENCE_LENGTH - 1
    FEATURE_COLUMNS = ['HS', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    TARGET_COLUMN = 'no_snow'
    DATE_COLUMN = 'measure_date'
    SPLIT_PERCENTAGE = 0.8
    DATASET_BATCH_SIZE = 64

    # Active learning
    UNCERTAINTY_ITERATIONS = 5

    # Model configuration
    MODEL_ARCHITECTURE = "128(l)-64-8(d)-1"
    MODEL_INPUT_SHAPE = (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    MODEL_DROPOUT_RATE = 0.5
    MODEL_OPTIMIZER = 'adam'
    MODEL_METRICS = ['accuracy']
    MODEL_LOSS = 'binary_crossentropy'
    MODEL_BATCH_SIZE = 64
    MODEL_EPOCHS = 10

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
            targets=None if targets is None else targets[self.TARGET_START_INDEX:],
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

    def get_most_uncertain_prediction(self, tasks):
        most_uncertain = {
            'predictions_score': 0
        }

        for task in tasks:
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

        # TODO: Replace localhost with environment variable
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

    def simulate_data_label(self, task):
        # TODO: Should be extracted into a util function and not be a part of the active learning loop because
        #  it is considered that a human expert will label the data
        task_id = task['id']
        station_code = task['data']['station_code']
        labeled_df = pd.read_csv(f'../data/labeled_daily/{station_code}.csv', index_col=False)
        # TODO: Remove this in favor of a unified generate_payload method
        labeled_df[self.DATE_COLUMN] = pd.to_datetime(labeled_df[self.DATE_COLUMN])
        payload = self.generate_payload(labeled_df)
        self.project.create_annotation(task_id, result=payload)

    @staticmethod
    def create_model(
            architecture=MODEL_ARCHITECTURE,
            input_shape=MODEL_INPUT_SHAPE,
            dropout_rate=MODEL_DROPOUT_RATE,
            summary=False
    ):
        arch_split = architecture.split('-')
        dense = True
        bidirectional = False
        layers = []

        digits_pattern = re.compile(r"\d+")

        if 'l' in arch_split[0]:
            rnn_layer = 'LSTM'
        elif 'g' in arch_split[0]:
            rnn_layer = 'GRU'
        else:
            raise Exception('rnn_layers must be one of [LSTM, GRU]')

        if 'b' in arch_split[0]:
            bidirectional = True

        for i, layer in enumerate(reversed(arch_split)):
            no_units = int(digits_pattern.findall(layer)[0])

            if dense:
                activation = 'sigmoid' if i == 0 else 'relu'
                layers.append(keras.layers.Dense(no_units, activation=activation))
            else:
                args = {
                    'units': no_units,
                }

                if '(d)' not in arch_split[-i]:
                    args['return_sequences'] = True

                if i == len(arch_split) - 1:
                    args['input_shape'] = input_shape

                current_layer = keras.layers.LSTM(**args) if rnn_layer == 'LSTM' else keras.layers.GRU(**args)

                if bidirectional:
                    current_layer = keras.layers.Bidirectional(current_layer)

                layers.extend([
                    MonteCarloDropout(dropout_rate),
                    current_layer
                ])

            if '(d)' in layer:
                dense = False

        layers.reverse()

        if summary:
            logging.info('---------- Model summary ----------')
            for layer in layers:
                layer_type = str(type(layer))

                if 'Dense' in str(type(layer)):
                    logging.info(type(layer), layer.units, layer.activation)
                elif 'Bidirectional' in str(type(layer)):
                    logging.info(type(layer), layer.layer, layer.layer.units, layer.layer.activation,
                                 layer.layer.return_sequences)
                elif 'LSTM' in str(type(layer)):
                    logging.info(type(layer), layer.units, layer.activation, layer.return_sequences)
                elif 'GRU' in str(type(layer)):
                    logging.info(type(layer), layer.units, layer.activation, layer.return_sequences)
                elif 'Dropout' in layer_type:
                    logging.info(type(layer), layer.rate)

        return keras.Sequential(layers)

    def predict(self, unlabeled_tasks, model):
        # tasks = self.project.get_paginated_tasks(page_size=10, page=1)['tasks']
        data = {}

        for task in unlabeled_tasks:
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
        # model = keras.models.load_model(
        #     '../active-learning-results/2023-11-02/model.keras',
        #     custom_objects={'MonteCarloDropout': MonteCarloDropout}
        # )

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

    def run_iteration(self):
        labeled_tasks = self.project.get_labeled_tasks(only_ids=True)
        unlabeled_tasks = self.project.get_unlabeled_tasks()
        # task = random.choice(unlabeled_tasks)

        # 1. Pick a random station to label if this is the first iteration; otherwise, choose the one with the
        # highest uncertainty score
        task = unlabeled_tasks[0] if len(labeled_tasks) == 0 else self.get_most_uncertain_prediction(unlabeled_tasks)

        # TODO: Might not work in the initial run
        logging.info(f"Most uncertain: {task['data']['station_code']} with a score of {task['predictions_score']:.2f}")

        # 2. Ask for labels.
        # TODO: This could potentially be a chrome notification or an email to annotators to label a specific station.

        # 3. Label data
        self.simulate_data_label(task)

        # 4. Parse and split labeled data. Refetch the tasks to include the newly labeled station data in the
        # train/validation dataset split
        labeled_tasks = self.project.get_labeled_tasks()
        train_dataset, val_dataset = self.create_train_val_datasets(labeled_tasks)

        # 5. Build model
        model = self.create_model()
        model.compile(
            optimizer=self.MODEL_OPTIMIZER,
            metrics=self.MODEL_METRICS,
            loss=self.MODEL_LOSS
        )
        model.summary()

        # 6. Fit model
        history = model.fit(
            train_dataset,
            epochs=self.MODEL_EPOCHS,
            batch_size=self.MODEL_BATCH_SIZE,
            validation_data=val_dataset
        )

        # 7. Predict and assign uncertainty scores for unlabeled tasks
        unlabeled_tasks = self.project.get_unlabeled_tasks()
        self.predict(unlabeled_tasks, model)

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
        # TODO: Replace localhost with env variable
        url = f'http://localhost:8080/api/dm/actions?id=delete_tasks_annotations&project={self.project_id}'

        response = requests.post(url=url, headers=self.headers)

        if response.status_code == 200:
            json_response = response.json()
            logging.info(f"Annotations were purged. Total: {json_response['processed_items']}")
        else:
            raise Exception(f'Failed to purge annotations: {response.status_code}')


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
    # active_learner.purge_annotations()
    active_learner.run_iteration()

import os
import re
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from dotenv import load_dotenv

from label_studio_client import LabelStudioClient
from utils import setup_logger, format_with_border, measure_execution_time, get_and_increment_iteration
import logging

import mlflow


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

    def __init__(
            self,
            base_url,
            api_key,
            project_name,
            labeled_train_data_path,
            labeled_test_data_path,
            mlflow_tracking_url,
            mlflow_experiment_name,
            logging_tmp_log_file
    ):
        super().__init__(base_url, api_key, project_name)
        self.labeled_train_data_path = labeled_train_data_path
        self.labeled_test_data_path = labeled_test_data_path
        self.logging_tmp_log_file = logging_tmp_log_file

        mlflow.set_tracking_uri(mlflow_tracking_url)
        mlflow.set_experiment(mlflow_experiment_name)
        mlflow.tensorflow.autolog()

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
        df['start_range'] = df[self.TARGET_COLUMN] & (~df[self.TARGET_COLUMN].shift(1, fill_value=False))
        df['end_range'] = df[self.TARGET_COLUMN] & (~df[self.TARGET_COLUMN].shift(-1, fill_value=False))

        start_dates = df[df['start_range']][self.DATE_COLUMN]
        end_dates = df[df['end_range']][self.DATE_COLUMN].reset_index(drop=True)

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

    @staticmethod
    def get_most_uncertain_prediction(tasks):
        most_uncertain = {
            'predictions_score': 0
        }

        for task in tasks:
            if task['predictions_score'] > most_uncertain['predictions_score']:
                most_uncertain = task

        return most_uncertain

    def parse_df(self, task):
        path = task['data']['csv']
        labels = task['annotations'][-1]['result']

        start_dates = []
        end_dates = []

        for label in labels:
            start_dates.append(pd.Timestamp(label['value']['start']))
            end_dates.append(pd.Timestamp(label['value']['end']))

        response = requests.get(url=f"{self.base_url}{path}", headers=self.headers)

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

        mlflow.log_param('al_training_samples', len(train_df))
        mlflow.log_param('al_validation_samples', len(val_df))

        features, targets, mean, std = self.get_features_and_targets(combined_df, split_index)

        train_dataset = self.create_dataset(features, targets, end_index=split_index, shuffle=True)
        val_dataset = self.create_dataset(features, targets, start_index=split_index, shuffle=True)

        return train_dataset, val_dataset, mean, std

    def simulate_data_label(self, task):
        task_id = task['id']
        station_code = task['data']['station_code']
        labeled_df = pd.read_csv(f'{self.labeled_train_data_path}/{station_code}.csv', index_col=False)
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
            logging.info(format_with_border('Model Summary'))
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
        data = {}

        for task in unlabeled_tasks:
            response = requests.get(url=f"{self.base_url}/{task['data']['csv']}", headers=self.headers)

            if response.status_code == 200:
                data[task['data']['station_code']] = {
                    'task_id': task['id'],
                    'csv': task['data']['csv'],
                    'df': pd.read_csv(BytesIO(response.content), encoding='UTF-8').dropna()
                }
            else:
                raise Exception(f'Failed to retrieve data: {response.status_code}')

        for k in data.keys():
            df = self.preprocces(data[k]['df'])
            features, targets, _, _ = self.get_features_and_targets(df, target_column=None, split_percentage=1)
            dataset = self.create_dataset(features, targets)

            predictions, uncertainty_score = self.predict_with_uncertainty(model, dataset)

            logging.info(f'Station: {k}, Uncertainty Score: {uncertainty_score:.4f}')

            df[self.TARGET_COLUMN] = False
            df.loc[self.TARGET_START_INDEX:, self.TARGET_COLUMN] = predictions > 0.5

            payload = self.generate_payload(df)
            self.project.create_prediction(task_id=data[k]['task_id'], result=payload, score=float(uncertainty_score))

    def run_iteration(self):
        iteration_start_time = time.time()
        iteration = get_and_increment_iteration()['iteration']

        with mlflow.start_run(run_name=f'Iteration {iteration}'):
            mlflow.log_param('al_iteration', iteration)
            logging.info(format_with_border(f'Iteration {iteration}'))

            labeled_tasks = self.project.get_labeled_tasks(only_ids=True)
            unlabeled_tasks = self.project.get_unlabeled_tasks()
            # task = random.choice(unlabeled_tasks)

            mlflow.log_param('al_labeled_tasks', len(labeled_tasks))
            mlflow.log_param('al_unlabeled_tasks', len(unlabeled_tasks))

            # 1. Pick a random station to label if this is the first iteration; otherwise, choose the one with the
            # highest uncertainty score
            task = unlabeled_tasks[0] if len(labeled_tasks) == 0 else self.get_most_uncertain_prediction(unlabeled_tasks)

            logging.info(f"Active Station: {task['data']['station_code']}")
            mlflow.log_param('al_active_station', task['data']['station_code'])

            # 2. Ask for labels.

            # 3. Label data
            self.simulate_data_label(task)

            # 4. Parse and split labeled data. Refetch the tasks to include the newly labeled station data in the
            # train/validation dataset split
            logging.info(format_with_border('Preparing Training Data'))
            labeled_tasks = self.project.get_labeled_tasks()
            stations = ', '.join(task['data']['station_code'] for task in labeled_tasks)
            logging.info(f'Stations: {stations}')
            mlflow.log_param('al_training_station_names', stations.split(', '))
            mlflow.log_param('al_training_split_percentage', self.SPLIT_PERCENTAGE)
            train_dataset, val_dataset, mean, std = self.create_train_val_datasets(labeled_tasks)

            # 5. Build model
            model = self.create_model()
            model.compile(
                optimizer=self.MODEL_OPTIMIZER,
                metrics=self.MODEL_METRICS,
                loss=self.MODEL_LOSS
            )

            # 6. Fit model
            logging.info(format_with_border('Training Model'))

            @measure_execution_time
            def fit_model():
                model.fit(
                    train_dataset,
                    epochs=self.MODEL_EPOCHS,
                    batch_size=self.MODEL_BATCH_SIZE,
                    validation_data=val_dataset
                )

            history, elapsed_fitting_time = fit_model()

            logging.info(f'Model fitting completed in {elapsed_fitting_time} minutes')
            mlflow.log_param('al_model_fitting_time', elapsed_fitting_time)

            # 7. Evaluate model on test data
            logging.info(format_with_border('Evaluating Model on Test Data'))
            files = os.listdir(self.labeled_test_data_path)

            all_evaluation_results = np.empty((0, 2), float)

            for i, f in enumerate(files):
                test_df = pd.read_csv(os.path.join(self.labeled_test_data_path, f), index_col=False)
                test_df = self.preprocces(test_df)
                station_name = test_df.iloc[0]['station_code']

                features, targets, _, _ = self.get_features_and_targets(test_df, split_percentage=1, mean=mean, std=std)
                test_dataset = self.create_dataset(features, targets)

                evaluation_results = model.evaluate(test_dataset, verbose=0)
                logging.info(f'Station: {station_name}, Samples: {len(test_df)}, Loss: {evaluation_results[0]:.2f}, Accuracy: {evaluation_results[1]:.2f}')

                # TODO: Remove this since nested experiment is removed
                # mlflow.log_param('al_test_station', station_name)
                # mlflow.log_param('al_test_test_samples', len(test_df))

                # mlflow.log_metric(f"test_{station_name}_loss", evaluation_results[0], step=iteration)
                # mlflow.log_metric(f"test_{station_name}_accuracy", evaluation_results[1], step=iteration)
                #
                all_evaluation_results = np.append(all_evaluation_results, [evaluation_results], axis=0)

            average_loss = np.mean(all_evaluation_results[:, 0])
            average_accuracy = np.mean(all_evaluation_results[:, 1])

            mlflow.log_metric('test_avg_loss', average_loss, step=iteration)
            mlflow.log_metric('test_avg_accuracy', average_accuracy, step=iteration)

            # 8. Predict and assign uncertainty scores for unlabeled tasks
            # TODO: Log uncertainty scores
            logging.info(format_with_border('Assigning Uncertainty Scores'))
            unlabeled_tasks = self.project.get_unlabeled_tasks()
            self.predict(unlabeled_tasks, model)

            logging.info(format_with_border('Done'))
            iteration_duration = time.time() - iteration_start_time
            minutes, seconds = divmod(iteration_duration, 60)
            formatted_time = f"{int(minutes)}:{int(seconds):02d}"
            logging.info(f'Iteration completed in {formatted_time} minutes')
            mlflow.log_param('al_iteration_completion_time', formatted_time)

            mlflow.log_artifact(tmp_log_file)

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
        logging.info(format_with_border('Purging Annotations'))
        url = f'{self.base_url}/api/dm/actions?id=delete_tasks_annotations&project={self.project_id}'

        response = requests.post(url=url, headers=self.headers)

        if response.status_code == 200:
            json_response = response.json()
            logging.info(f"{json_response['processed_items']} annotation(s) have been purged")
        else:
            raise Exception(f'Failed to purge annotations: {response.status_code}')


if __name__ == '__main__':
    load_dotenv()
    logging, tmp_log_file = setup_logger(log_file='active_learning_run.log')

    BASE_URL = os.getenv('LS_BASE_URL')
    API_KEY = os.getenv('LS_API_KEY')
    PROJECT_NAME = os.getenv('LS_PROJECT_NAME')

    LABELED_TRAIN_DATA_PATH = os.getenv('LS_LABELED_TRAIN_DATA_PATH')
    LABELED_TEST_DATA_PATH = os.getenv('LS_LABELED_TEST_DATA_PATH')

    MLFLOW_PORT = os.getenv('MLFLOW_PORT')
    MLFLOW_URI = f'http://localhost:{MLFLOW_PORT}'

    active_learner = ActiveLearner(
        base_url=BASE_URL,
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        labeled_train_data_path=LABELED_TRAIN_DATA_PATH,
        labeled_test_data_path=LABELED_TEST_DATA_PATH,
        mlflow_tracking_url=MLFLOW_URI,
        mlflow_experiment_name="No Snow Classification 3",
        logging_tmp_log_file=tmp_log_file
    )

    # Set initial predictions
    active_learner.purge_annotations()
    active_learner.run_iteration()

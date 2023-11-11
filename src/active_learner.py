import os
from io import BytesIO

from src.active_learning_iteration import ActiveLearningIteration

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.functions import create_model, preprocces, create_train_val_datasets, create_test_dataset, \
    predict_with_uncertainty
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from dotenv import load_dotenv

from label_studio_client import LabelStudioClient
from utils import setup_logger, format_with_border, measure_execution_time

import mlflow


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
    # MODEL_ARCHITECTURE = "128(l)-64-8(d)-1"
    MODEL_ARCHITECTURE = "32(l)-8(d)-1"
    MODEL_INPUT_SHAPE = (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    MODEL_DROPOUT_RATE = 0.5
    MODEL_OPTIMIZER = 'adam'
    MODEL_METRICS = ['accuracy']
    MODEL_LOSS = 'binary_crossentropy'
    MODEL_BATCH_SIZE = 64
    MODEL_EPOCHS = 3

    def __init__(
            self,
            base_url,
            api_key,
            project_name,
            labeled_train_data_path,
            labeled_test_data_path,
            mlflow_tracking_url,
            mlflow_experiment_name,
            log_file_path,
            iteration
    ):
        super().__init__(base_url, api_key, project_name)
        self.labeled_train_data_path = labeled_train_data_path
        self.labeled_test_data_path = labeled_test_data_path
        self.log_file_path = log_file_path
        self.iteration: ActiveLearningIteration = iteration

        self._logger = None

        mlflow.set_tracking_uri(mlflow_tracking_url)
        mlflow.set_experiment(mlflow_experiment_name)
        mlflow.tensorflow.autolog()

    # Delay the initialization of the logger to prevent overwriting the log file when trying to run multiple experiments
    # concurrently
    @property
    def logger(self):
        if self._logger is None:
            self._logger, _ = setup_logger(log_file_path=self.log_file_path)

        return self._logger

    def log_parameters(self, iteration):
        for attr in dir(self):
            if attr.isupper():
                value = getattr(self, attr)
                self.logger.info(f'{attr}: {value}')
                mlflow.log_param(f'al_{attr.lower()}', value)

        mlflow.log_param('al_iteration', iteration)

    def generate_prediction_payload(self, df):
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

    def simulate_data_label(self, task):
        station_code = task['data']['station_code']
        labeled_df = pd.read_csv(f'{self.labeled_train_data_path}/{station_code}.csv', index_col=False)
        labeled_df[self.DATE_COLUMN] = pd.to_datetime(labeled_df[self.DATE_COLUMN])

        return labeled_df

    def predict(self, unlabeled_tasks, model, mean, std):
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
            df = preprocces(data[k]['df'])

            dataset = create_test_dataset(
                df,
                self.FEATURE_COLUMNS,
                None,
                self.SEQUENCE_LENGTH,
                self.TARGET_START_INDEX,
                self.DATASET_BATCH_SIZE,
                mean,
                std
            )

            predictions, uncertainty_score = predict_with_uncertainty(model, dataset,
                                                                      n_iter=self.UNCERTAINTY_ITERATIONS)

            self.logger.info(f'Station: {k}, Uncertainty Score: {uncertainty_score:.4f}')

            df[self.TARGET_COLUMN] = False
            df.loc[self.TARGET_START_INDEX:, self.TARGET_COLUMN] = predictions > 0.5

            payload = self.generate_prediction_payload(df)
            self.project.create_prediction(task_id=data[k]['task_id'], result=payload, score=float(uncertainty_score))

    def run_iteration(self):
        if self.iteration.is_locked():
            raise RuntimeError(f"An iteration is already running. Please try again later.")

        iteration_start_time = time.time()
        iteration = self.iteration.get_and_increment()
        self.iteration.lock()
        with mlflow.start_run(run_id=iteration['run_id'], run_name=f"Run_{iteration['run_name']}") as run:
            self.iteration.set_run_id(run.info.run_id)
            with mlflow.start_run(run_name=f"Iteration {iteration['iteration']}", nested=True):
                self.logger.info(format_with_border(f"Iteration {iteration['iteration']}"))
                self.log_parameters(iteration['iteration'])

                labeled_tasks = self.project.get_labeled_tasks(only_ids=True)
                unlabeled_tasks = self.project.get_unlabeled_tasks()
                # task = random.choice(unlabeled_tasks)

                mlflow.log_param('al_labeled_tasks', len(labeled_tasks))
                mlflow.log_param('al_unlabeled_tasks', len(unlabeled_tasks))

                # 1. Pick a random station to label if this is the first iteration; otherwise, choose the one with the
                # highest uncertainty score
                task = unlabeled_tasks[0] if len(labeled_tasks) == 0 else self.get_most_uncertain_prediction(
                    unlabeled_tasks)

                self.logger.info(f"Active Station: {task['data']['station_code']}")
                mlflow.log_param('al_active_station', task['data']['station_code'])

                # 2. Ask for labels.

                # 3. Label data
                payload = self.generate_prediction_payload(self.simulate_data_label(task))
                self.project.create_annotation(task['id'], result=payload)

                # 4. Parse and split labeled data. Refetch the tasks to include the newly labeled station data in the
                # train/validation dataset split
                self.logger.info(format_with_border('Preparing Training Data'))
                labeled_tasks = self.project.get_labeled_tasks()
                stations = ', '.join(task['data']['station_code'] for task in labeled_tasks)
                self.logger.info(f'Stations: {stations}')
                mlflow.log_param('al_training_station_names', stations)
                parsed_labeled_tasks = [self.parse_df(task) for task in labeled_tasks]
                train_dataset, val_dataset, mean, std, num_train_samples, num_val_samples, _ = create_train_val_datasets(
                    parsed_labeled_tasks,
                    self.SPLIT_PERCENTAGE,
                    self.FEATURE_COLUMNS,
                    self.TARGET_COLUMN,
                    self.SEQUENCE_LENGTH,
                    self.TARGET_START_INDEX,
                    self.DATASET_BATCH_SIZE
                )
                self.logger.info(f"Training samples: {num_train_samples}")
                self.logger.info(f"Validation samples: {num_val_samples}")
                mlflow.log_param('al_training_samples', num_train_samples)
                mlflow.log_param('al_validation_samples', num_val_samples)

                # 5. Build and compile model
                model = create_model(
                    architecture=self.MODEL_ARCHITECTURE,
                    input_shape=self.MODEL_INPUT_SHAPE,
                    dropout_rate=self.MODEL_DROPOUT_RATE,
                    logging=self.logger,
                    summary=False
                )
                model.compile(
                    optimizer=self.MODEL_OPTIMIZER, metrics=self.MODEL_METRICS, loss=self.MODEL_LOSS
                )

                # 6. Fit model
                self.logger.info(format_with_border('Fitting Model'))

                @measure_execution_time
                def fit_model():
                    model.fit(
                        train_dataset,
                        epochs=self.MODEL_EPOCHS,
                        batch_size=self.MODEL_BATCH_SIZE,
                        validation_data=val_dataset
                    )

                history, elapsed_fitting_time = fit_model()
                self.logger.info(f'Model fitting completed in {elapsed_fitting_time}')
                mlflow.log_param('al_model_fitting_time', elapsed_fitting_time)

                # 7. Evaluate model on test data
                self.logger.info(format_with_border('Evaluating Model on Test Data'))
                average_accuracy, average_loss = self.evaluate_on_test_data(model, mean, std)
                mlflow.log_metric('test_avg_loss', average_loss)
                mlflow.log_metric('test_avg_accuracy', average_accuracy)

                # 8. Predict and assign uncertainty scores for unlabeled tasks
                self.logger.info(format_with_border('Assigning Uncertainty Scores'))
                unlabeled_tasks = self.project.get_unlabeled_tasks()
                self.predict(unlabeled_tasks, model, mean, std)

                # 9. Done
                self.logger.info(format_with_border('Done'))
                iteration_duration = time.time() - iteration_start_time
                minutes, seconds = divmod(iteration_duration, 60)
                formatted_time = f"{int(minutes)}m{int(seconds)}s"
                self.logger.info(f'Iteration completed in {formatted_time}')
                mlflow.log_param('al_iteration_completion_time', formatted_time)

                mlflow.log_artifact(self.log_file_path)
        self.iteration.unlock()

    def evaluate_on_test_data(self, model, mean, std):
        files = os.listdir(self.labeled_test_data_path)
        all_evaluation_results = np.empty((0, 2), float)
        for i, f in enumerate(files):
            test_df = pd.read_csv(os.path.join(self.labeled_test_data_path, f), index_col=False)
            test_df = preprocces(test_df)
            station_name = test_df.iloc[0]['station_code']

            test_dataset = create_test_dataset(
                test_df,
                self.FEATURE_COLUMNS,
                self.TARGET_COLUMN,
                self.SEQUENCE_LENGTH,
                self.TARGET_START_INDEX,
                self.DATASET_BATCH_SIZE,
                mean,
                std
            )

            evaluation_results = model.evaluate(test_dataset, verbose=0)
            self.logger.info(
                f'Station: {station_name}, Samples: {len(test_df)}, Loss: {evaluation_results[0]:.2f}, Accuracy: {evaluation_results[1]:.2f}')

            all_evaluation_results = np.append(all_evaluation_results, [evaluation_results], axis=0)
        average_loss = np.mean(all_evaluation_results[:, 0])
        average_accuracy = np.mean(all_evaluation_results[:, 1])
        return average_accuracy, average_loss

    def display_time_series(self, task):
        df = self.parse_df(task)
        fig = px.line(df, y="HS", title="Time Series with Outliers")
        fig.add_trace(go.Scatter(
            x=df[df['outlier'] == 1][self.DATE_COLUMN],
            y=df[df['outlier'] == 1]['HS'],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=8)
        ))
        fig.show()

    def purge_annotations(self):
        url = f'{self.base_url}/api/dm/actions?id=delete_tasks_annotations&project={self.project_id}'

        response = requests.post(url=url, headers=self.headers)

        if response.status_code == 200:
            json_response = response.json()
            self.logger.info(f"{json_response['processed_items']} annotation(s) were purged")
        else:
            raise Exception(f'Failed to purge annotations: {response.status_code}')

    def purge_predictions(self):
        url = f'{self.base_url}/api/dm/actions?id=delete_tasks_predictions&project={self.project_id}'

        response = requests.post(url=url, headers=self.headers)

        if response.status_code == 200:
            json_response = response.json()
            self.logger.info(f"{json_response['processed_items']} prediction(s) were purged")
        else:
            raise Exception(f'Failed to purge predictions: {response.status_code}')

    def reset(self):
        self.logger.info(format_with_border('Resetting environment'))
        self.purge_annotations()
        self.purge_predictions()
        self.iteration.reset()

    @staticmethod
    def build():
        load_dotenv()
        LOG_FILE_PATH = os.getenv('AL_LOG_FILE_PATH')

        BASE_URL = os.getenv('LS_BASE_URL')
        API_KEY = os.getenv('LS_API_KEY')
        PROJECT_NAME = os.getenv('LS_PROJECT_NAME')

        LABELED_TRAIN_DATA_PATH = os.getenv('LS_LABELED_TRAIN_DATA_PATH')
        LABELED_TEST_DATA_PATH = os.getenv('LS_LABELED_TEST_DATA_PATH')

        MLFLOW_PORT = os.getenv('MLFLOW_PORT')
        MLFLOW_URI = f'http://localhost:{MLFLOW_PORT}'

        return ActiveLearner(
            base_url=BASE_URL,
            api_key=API_KEY,
            project_name=PROJECT_NAME,
            labeled_train_data_path=LABELED_TRAIN_DATA_PATH,
            labeled_test_data_path=LABELED_TEST_DATA_PATH,
            mlflow_tracking_url=MLFLOW_URI,
            mlflow_experiment_name="Snow Height Anomaly Detection",
            log_file_path=LOG_FILE_PATH,
            iteration=ActiveLearningIteration(iteration_file_path='../active-learning.json')
        )


if __name__ == '__main__':
    active_learner = ActiveLearner.build()

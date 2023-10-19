import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from pyod.models.iforest import IForest

from label_studio_client import LabelStudioClient
from utils import setup_logger


class ActiveLearner(LabelStudioClient):
    def __init__(self, base_url, api_key, project_name):
        super().__init__(base_url, api_key, project_name)

    def predict(self):
        # tasks = self.project.get_paginated_tasks(page_size=10, page=1)['tasks']
        tasks = self.project.get_tasks()

        data = {}

        for task in tasks:
            data[task['data']['station_code']] = {
                'task_id': task['id'],
                'csv': task['data']['csv'],
            }

        df = pd.concat((pd.read_csv(data[station]['csv']) for station in data), ignore_index=True).dropna()

        X = df.drop(columns=['measure_date', 'station_code', 'HN_1D'])

        # start with initial model
        iforest = IForest(random_state=42)
        iforest.fit(X.values)
        probs = iforest.predict_proba(X.values)

        df['is_outlier'] = probs[:, 1] >= 0.60
        df['score'] = probs[np.arange(len(df)), df['is_outlier'].astype(int)]

        for station, group in df[df['is_outlier']].groupby('station_code'):
            payload = [
                {
                    "type": "timeserieslabels",
                    "value": {
                        "end": row['measure_date'],
                        "start": row['measure_date'],
                        "instant": True,
                        "timeserieslabels": ["Outlier"]
                    },
                    "to_name": "ts",
                    "from_name": "label"
                }
                for (_, row) in group.iterrows()
            ]

            self.project.create_prediction(task_id=data[station]['task_id'], result=payload, score=group['score'].mean())

    def test(self):
        tasks = self.project.get_labeled_tasks()

        path = tasks[0]['data']['csv']
        labels = tasks[0]['annotations'][0]['result']

        start_dates = []
        end_dates = []

        for label in labels:
            start_dates.append(pd.Timestamp(label['value']['start']))
            end_dates.append(pd.Timestamp(label['value']['end']))

        df = pd.read_csv(path)
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

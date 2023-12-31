import json
import os
from datetime import datetime

from src.utils import now_formatted

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging


class ActiveLearningIteration:
    def __init__(self, iteration_file_path):
        self.iteration_file_path = iteration_file_path

    def get(self):
        default_data = {
            'iteration': 0,
            'last_execution_date': None,
            'run_name': now_formatted(),
            'run_id': None,
            'locked': False
        }
        try:
            if not os.path.exists(self.iteration_file_path):
                return default_data
            with open(self.iteration_file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Error encountered: {e}")
            return default_data

    def get_and_increment(self):
        iteration = self.get()
        updated_iteration = iteration.copy()
        updated_iteration['iteration'] += 1
        updated_iteration['last_execution_date'] = now_formatted()

        self.persist(updated_iteration)

        return iteration

    def set_run_id(self, run_id):
        iteration = self.get()
        iteration['run_id'] = run_id
        self.persist(iteration)

    def persist(self, iteration):
        with open(self.iteration_file_path, 'a+') as file:
            file.seek(0)
            file.truncate()
            json.dump(iteration, file)

    def reset(self):
        try:
            os.remove(self.iteration_file_path)
            logging.info(f'Iteration file has been reset')
        except FileNotFoundError:
            logging.error(f'The file {self.iteration_file_path} does not exist')

    def lock(self):
        iteration = self.get()
        iteration['locked'] = True
        self.persist(iteration)

    def unlock(self):
        iteration = self.get()
        iteration['locked'] = False
        self.persist(iteration)

    def is_locked(self):
        return self.get()['locked']

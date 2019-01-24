import pandas as pd


class DataInit():
    def __init__(self, file_path):
        data = self.load_data(file_path)

        # data = self.clean_data(data)

        data = self.preprocess_data(data['question1'], data['question2'])

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def preprocess_data(self, q1, q2):
        pass
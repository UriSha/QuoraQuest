import pandas as pd
import re

class DataInit():
    def __init__(self, file_path):
        data = self.load_data(file_path)

        data = self.clean_data(data)

        data = self.preprocess_data(data['question1'], data['question2'])

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def preprocess_data(self, q1, q2):
        pass

    def clean_data(self, data):
        data['question1'] = data.apply(lambda row: self.normalize_string(row['question1']), axis=1)
        data['question2'] = data.apply(lambda row: self.normalize_string(row['question2']), axis=1)

    def normalize_string(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

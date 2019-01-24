import pandas as pd
import re

class DataInit():
    def __init__(self, file_path):
        data_frame = self.load_data(file_path)
        print(data_frame.dtypes)

        data_frame = self.clean_data(data_frame)

        self.data = self.preprocess_data(data_frame['question1'], data_frame['question2'])

        self.labels = data_frame['is_duplicate'].values

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def preprocess_data(self, q1, q2):
        res = []
        for q1, q2 in zip(q1, q2):
            res.append(q1)
            res.append(q2)
        return res

    def clean_data(self, data):
        data['question1']= data['question1'].astype(str)
        data['question2']= data['question2'].astype(str)
        data['question1'] = data.apply(lambda row: self.normalize_string(row['question1']), axis=1)
        data['question2'] = data.apply(lambda row: self.normalize_string(row['question2']), axis=1)

    def normalize_string(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def get_data_and_labels(self):
        return self.data, self.labels

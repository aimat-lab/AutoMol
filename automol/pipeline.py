from automol.datasets import Dataset
from automol.models import ModelGenerator

import yaml
import requests
import numpy

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow


class Pipeline:

    def __init__(self, input_file):
        with open(input_file, 'r') as file:
            try:
                self.spec = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise e
        self.data_set = Dataset.from_spec(self.spec)
        self.model_generator = ModelGenerator()
        self.models = []

        self.custom_features = self.parse_custom_features(self.spec['custom_features'])
        self.data_set.feature_generator().add_custom_features(self.custom_features)

    def train(self, test_size=0.25):
        mlflow.sklearn.autolog()
        index_split = int((1. - test_size) * self.data_set.data.shape[0])
        train, test = self.data_set.split(index_split)

        for model in self.model_generator.generate_all_possible_models(
                self.data_set, self.spec['problem'], self.spec['models_filter']):
            self.models.append(model)
            with mlflow.start_run():
                model.fit(train, self.spec['labels'])
            self.print_statistics(model, test)

    def print_statistics(self, model, test):
        stats = self.get_statistics(model, test)
        print("Model '%s' with feature '%s' has MAE: %f" % (model, model.feature_name, stats['mae']))
        print("Model '%s' with feature '%s' has MSE: %f" % (model, model.feature_name, stats['mse']))
        print("Model '%s' with feature '%s' has R2S: %f" % (model, model.feature_name, stats['r2s']))

    def get_statistics(self, model, test):
        pred = model.predict(test)
        y_test = test[self.spec['labels']]
        return {
            'mae': mean_absolute_error(y_test, pred),
            'mse': mean_squared_error(y_test, pred),
            'r2s': r2_score(y_test, pred),
        }

    @staticmethod
    def parse_custom_features(custom_features):
        r = {}
        for k, v in custom_features.items():
            ns = {}
            exec(requests.get(v['file_link']).text, ns)

            def a(data_set, func=ns[v['function_name']]):
                data = numpy.array(
                    [data_set.feature_generator().get_feature(param_name) for param_name in v['input']]).transpose()
                return numpy.array([func(*data[i]) for i in data_set.data.index])
            r[k] = {
                'iam': set(v['iam']),
                'requirements': v['input'],
                'transform': a
            }
        return r

    @staticmethod
    def parse_custom_features2(custom_features):
        r = {}
        for k, v in custom_features.items():
            file_link = v["file_link"]
            try:
                response = requests.get(file_link)
                response.raise_for_status()
            except requests.exceptions.HTTPError as errh:
                print(f"Http Error: {errh}")
            except requests.exceptions.ConnectionError as errc:
                print(f"Error Connecting: {errc}")
            except requests.exceptions.Timeout as errt:
                print(f"Timeout Error: {errt}")
            except requests.exceptions.RequestException as err:
                print(f"General error {err}")
            response_text = response.text

            def a(data_set, func=ns[v['function_name']]):
                data = numpy.array(
                    [data_set.feature_generator().get_feature(param_name) for param_name in v['input']]).transpose()
                return numpy.array([func(*data[i]) for i in data_set.data.index])

            r[k] = {
                'iam': set(v['iam']),
                'requirements': v['input'],
                'transform': a
            }
        return r

    def print_spec(self):
        print(yaml.dump(self.spec))


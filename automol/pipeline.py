from automol.datasets import Dataset
from automol.features import FeatureGenerator
from automol.models import ModelGenerator
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Pipeline:

    def __init__(self, input_file):
        with open(input_file, 'r') as file:
            try:
                self.spec = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise e
        self.data_set = Dataset.from_spec(self.spec)
        self.feature_generator = FeatureGenerator(self.data_set)
        self.model_generator = ModelGenerator(self.feature_generator)
        self.models = []

    def train(self, y_test_size=0.25):
        y = self.data_set.data[self.spec['labels']]
        index_split = int((1. - y_test_size) * len(y))
        y_train, y_test = np.split(y, [0, index_split])
        for model in self.model_generator.get_models(self.spec['problem'], self.spec['models_to_exclude']):
            self.models.append(model)
            model.fit(y_train)
            pred = model.predict()
            print("Model %s has MAE: %f" % (model, mean_absolute_error(y_test, pred)))
            print("Model %s has MSE: %f" % (model, mean_squared_error(y_test, pred)))
            print("Model %s has R2S: %f" % (model, r2_score(y_test, pred)))

    def get_statistics(self):
        pass  # something here

    def print_spec(self):
        print(yaml.dump(self.spec))

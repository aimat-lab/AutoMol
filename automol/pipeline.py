

from .models import *


import yaml


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

    def train(self):
        for model in self.model_generator.generate_models(self.spec['models']):
            for label in self.data_set.data[self.spec['labels']]:
                X_feature_name = 'fingerprint'
                X = self.feature_generator.get_feature(X_feature_name)
                y = self.data_set.data[label]
                print("Training model {} with X = {} and y = {}".format(model, X_feature_name, label))
                model.fit(X, y)

    def get_statistics(self):
        pass  # something here

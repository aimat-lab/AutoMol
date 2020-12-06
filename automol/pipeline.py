

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
            model.train()

    def get_statistics(self):
        pass  # something here

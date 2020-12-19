from automol.features import FeatureGenerator
from automol.models import ModelGenerator

import yaml

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error


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

    def train(self):
        train, test, = train_test_split(self.data_set.data[self.spec['labels']], test_size=.25)
        for model in self.model_generator.get_models(self.spec['problem'], self.spec['models_to_exclude']):
            self.models.append(model)
            model.fit(train)
            pred = model.predict()
            print("Model %s has MAE: %f" % (model, pred))
            print("Model %s has MSE: %f" % (model, pred))

    def get_statistics(self):
        pass  # something here

    def print_spec(self):
        print(yaml.dump(self.spec))

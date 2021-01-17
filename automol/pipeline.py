from automol.datasets import Dataset
from automol.features import FeatureGenerator
from automol.models import ModelGenerator
import numpy
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
        self.model_generator = ModelGenerator()
        self.models = []

    def train(self, test_size=0.25):
        index_split = int((1. - test_size) * self.data_set.data.shape[0])
        train, test = self.data_set.split(index_split)
        y_test = test[self.spec['labels']]
        for model in self.model_generator.generate_all_possible_models(self.data_set, self.spec['problem'], self.spec['models_filter']):
            self.models.append(model)
            model.fit(train, self.spec['labels'])
            pred = model.predict(test)
            print("Model %s has MAE: %f" % (model, mean_absolute_error(y_test, pred)))
            print("Model %s has MSE: %f" % (model, mean_squared_error(y_test, pred)))
            print("Model %s has R2S: %f" % (model, r2_score(y_test, pred)))

    def get_statistics(self):
        # todo something here
        pass

    def print_spec(self):
        print(yaml.dump(self.spec))

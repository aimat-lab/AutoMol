from automol.features import FeatureGenerator
from automol.models import ModelGenerator
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
        for model in self.model_generator.get_models(self.spec['problem'], self.spec['models_to_exclude']):
            model.fit(self.data_set.data[self.spec['labels']])

    def get_statistics(self):
        pass  # something here

    def print_spec(self):
        print(yaml.dump(self.spec))

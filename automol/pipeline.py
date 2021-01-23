from automol.datasets import Dataset
from automol.models import ModelGenerator
import yaml
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

    def print_spec(self):
        print(yaml.dump(self.spec))

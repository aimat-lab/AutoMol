from automol.datasets import Dataset, DataSplit
from automol.models import ModelGenerator

import yaml
import requests
import numpy
import pandas

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

    def train(self):
        mlflow.sklearn.autolog()
        split_params = self.spec['dataset_split']['params']

        # creates the generator for the ith nested level of the iteration and progresses it
        def i_level(i):
            split_gen = iter(DataSplit.invoke(self.data_set if i == 0 else sets[i],
                                              self.spec['dataset_split']['method'],
                                              split_params[i]))
            if len(split_gens) > i:
                raise Exception('illegal')
            elif len(split_gens) == i:
                split_gens.append(split_gen)
            else:
                split_gens[i] = split_gen

        # progresses the ith nested level by 1 iteration
        def progr(i):
            sets[i], sets[i + 1] = next(split_gens[i])

        split_gens = []
        # k nested levels -> k splits -> (k + 1) sets
        sets = [pandas.DataFrame()] * (len(split_params) + 1)
        i_level(0)

        while split_gens:
            while split_gens:
                try:
                    progr(len(split_gens) - 1)
                    # all levels have been generated
                    if len(split_gens) == len(split_params):
                        break
                    else:
                        # generates the next level based on the new split
                        i_level(len(split_gens))
                except StopIteration:
                    del split_gens[-1]

            # FUTURE: might implement something actually useful here, like hyperparam search
            # rn: just prints the stats evaluated on each lower level set, uses topmost set to fit
            for model in self.model_generator.generate_all_possible_models(
                    self.data_set, self.spec['problem'], self.spec['models_filter']):
                self.models.append(model)
                with mlflow.start_run():
                    model.fit(sets[-1], self.spec['labels'])
                for i in range(len(sets) - 1):
                    print('stats on layer %d split:' % i)
                    self.print_statistics(model, sets[i])

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

    def print_spec(self):
        print(yaml.dump(self.spec))

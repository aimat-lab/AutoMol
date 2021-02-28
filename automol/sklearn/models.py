from typing import Set

import mlflow
import numpy

from automol.features.feature_generators import FeatureGenerator
from automol.models import Model

from sklearn.ensemble import *  # noqa
from sklearn.gaussian_process import *  # noqa
from sklearn.linear_model import *  # noqa
from sklearn.neural_network import *  # noqa

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa


def hyperparameter_search(model_name, feature_generator):
    return {
        'RandomForestRegressor': {
            'n_estimators': 42,
        },
    }.get(model_name, {})


class SklearnModelGenerator:
    __modelList = {}
    __modelTypes = {'Regressor': 'regression', 'Regression': 'regression', 'Classifier': 'classification'}

    for gvar, gval in globals().items():
        if issubclass(type(gval), type):
            for modelType in __modelTypes:
                if gvar.endswith(modelType):
                    if __modelTypes[modelType] not in __modelList:
                        __modelList[__modelTypes[modelType]] = {}
                    __modelList[__modelTypes[modelType]][gvar] = gval
                    break

    def __init__(self):
        pass

    def generate_all_possible_models(self, data_set, problem_type, models_filter=None):
        acceptable_model_names = self.__modelList[problem_type].keys()

        if models_filter is not None:
            acceptable_model_names = acceptable_model_names & set(models_filter['model_names']) \
                if models_filter['whitelist'] \
                else acceptable_model_names - set(models_filter['model_names'])

        return self.generate_models(data_set, problem_type, acceptable_model_names)

    def generate_models(self, data_set, problem_type, model_names):
        return [SklearnModelGenerator.generate_model(data_set, problem_type, model_name, acceptable_feature_gen)
                for model_name in model_names for acceptable_feature_gen in
                data_set.get_acceptable_feature_gens(self.acceptable_feature_types(model_name))]

    @staticmethod
    def generate_model(data_set, problem_type, model_name, feature_gen):
        type_list = SklearnModelGenerator.__modelList[problem_type]
        if model_name not in type_list:
            raise Exception('unknown model %s' % model_name)
        return SklearnModel(type_list[model_name](**hyperparameter_search(model_name, data_set.feature_generator())),
                            feature_gen)

    def get_model_type(self, model_name):
        return [modelType for modelType in self.__modelTypes if model_name.endswith(modelType)][0]

    def get_model_prefix(self, model_name):
        return str.replace(model_name, self.get_model_type(model_name), '')

    def acceptable_feature_types(self, model_name) -> Set[str]:
        return {
            'MLP': {'vector'},
            'Linear': {'vector'},
            'GaussianProcess': {'vector'},
            'GradientBoosting': {'vector'},
        }.get(self.get_model_prefix(model_name), set())


class SklearnModel(Model):

    def __init__(self, core, feature_gen: FeatureGenerator):
        self.core = core
        self.feature_gen: FeatureGenerator = feature_gen

    def run(self, sets, labels):
        with mlflow.start_run():
            self.core.fit(sets[-1], labels)
        for i in range(len(sets) - 1):
            print('stats on layer %d split:' % i)
            self.print_statistics(self.core, sets[i], labels)

    def print_statistics(self, model, test, labels):
        stats = self.get_statistics(model, test, labels)
        for k, v in stats.items():
            mlflow.log_metric(k, v)

    def get_statistics(self, model, test, labels):
        pred = model.predict(test)
        y_test = test[labels]
        return {
            'mae': mean_absolute_error(y_test, pred),
            'mse': mean_squared_error(y_test, pred),
            'r2s': r2_score(y_test, pred),
        }

    def fit(self, data_set, labels):
        inputs = self.feature_gen.transform(data_set.data)
        labels = numpy.array(data_set[labels]).flatten()
        # print(type(self.core))
        # print(labels.shape, labels)
        self.core.fit(inputs, labels)

    def predict(self, data_set):
        return self.core.predict(self.feature_gen.transform(data_set.data))

    def __str__(self):
        return self.core.__str__().replace("()", "")

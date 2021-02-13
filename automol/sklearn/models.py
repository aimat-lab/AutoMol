from typing import Set
from automol.models import Model

from sklearn.ensemble import * # noqa
from sklearn.gaussian_process import * # noqa
from sklearn.linear_model import * # noqa
from sklearn.neural_network import * # noqa

from sklearn.metrics import mean_absolute_error, mean_squared_error # noqa


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
            acceptable_model_names = acceptable_model_names & set(models_filter['model_names'])\
                if models_filter['whitelist']\
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
        return Model(type_list[model_name](**hyperparameter_search(model_name, data_set.feature_generator())),
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

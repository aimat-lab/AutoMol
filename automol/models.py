import numpy

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


class ModelGenerator:

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
        print(self.__modelList)
        pass

    def generate_all_possible_models(self, data_set, problem_type, models_filter=None):
        acceptable_model_names = self.__modelList[problem_type].keys()
        # whitelist/blacklist
        if models_filter is not None:
            acceptable_model_names = models_filter[1] if models_filter[0] == 'w' \
                else acceptable_model_names - set(models_filter[1])

        return self.generate_models(data_set, problem_type, acceptable_model_names)

    def generate_models(self, data_set, problem_type, model_names):
        return [ModelGenerator.generate_model(data_set, problem_type, model_name, acceptable_feature_name)
                for model_name in model_names for acceptable_feature_name in
                data_set.feature_generator().get_acceptable_features(self.acceptable_feature_types(model_name))]

    @staticmethod
    def generate_model(data_set, problem_type, model_name, feature_name):
        type_list = ModelGenerator.__modelList[problem_type]
        if model_name not in type_list:
            raise Exception('unknown model %s' % model_name)
        return Model(type_list[model_name](**hyperparameter_search(model_name, data_set.feature_generator())),
                     feature_name)

    def get_model_type(self, model_name):
        return [modelType for modelType in self.__modelTypes if model_name.endswith(modelType)][0]

    def get_model_prefix(self, model_name):
        return str.replace(model_name, self.get_model_type(model_name), '')

    def acceptable_feature_types(self, model_name):
        return {
            'MLP': {'vector'},
            'Linear': {'vector'},
            'GaussianProcess': {'vector'},
            'GradientBoosting': {'vector'},
        }.get(self.get_model_prefix(model_name), set())


class Model:

    def __init__(self, core, feature_name):
        self.core = core
        self.feature_name = feature_name

    def fit(self, data_set, labels):
        inputs = data_set.get_feature(self.feature_name)
        labels = numpy.array(data_set[labels]).flatten()
        # print(type(self.core))
        # print(labels.shape, labels)
        self.core.fit(inputs, labels)

    def predict(self, data_set):
        return self.core.predict(data_set.get_feature(self.feature_name))

    def __str__(self):
        return self.core.__str__().replace("()", "")

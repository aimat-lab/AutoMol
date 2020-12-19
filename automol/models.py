from automol.features import FeatureGenerator

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # noqa
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier # noqa
from sklearn.linear_model import LinearRegression, SGDClassifier # noqa
from sklearn.neural_network import MLPRegressor, MLPClassifier # noqa

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
            print(gvar)
            for modelType in __modelTypes:
                print(modelType, gvar.endswith(modelType))
                if gvar.endswith(modelType):
                    if modelType not in __modelList:
                        __modelList[__modelTypes[modelType]] = {}
                    __modelList[__modelTypes[modelType]][gvar] = gval
                    print(gvar, __modelList)
                    break

    def __init__(self, feature_generator: FeatureGenerator):
        self.feature_generator = feature_generator

    def get_models(self, problem_type, to_exclude=None):
        return self.generate_models(problem_type, self.__modelList[problem_type].keys() - to_exclude)

    def generate_models(self, problem_type, model_list):
        return [a for model_name in model_list for a in self.generate_model(problem_type, model_name)]

    def generate_model(self, problem_type, model_name):
        type_list = ModelGenerator.__modelList[problem_type]
        if model_name not in type_list:
            raise Exception('unknown model %s' % model_name)

        return [
            ModelAbstraction(type_list[model_name](**hyperparameter_search(model_name, self.feature_generator)),
                             feature)
            for feature in self.feature_generator.get_acceptable_features(self.acceptable_feature_types(model_name))
        ]

    def get_model_type(self, model_name):
        return [modelType for modelType in self.__modelTypes if model_name.endswith(modelType)][0]

    def get_model_prefix(self, model_name):
        return str.replace(model_name, self.get_model_type(model_name), '')

    def acceptable_feature_types(self, model_name):
        return {
            'MLP': {'vector'},
            'RandomForest': {'vector'},
            # todo add rest
        }.get(self.get_model_prefix(model_name), set())


class ModelAbstraction:
    def __init__(self, model, feature):
        self.model = model
        self.feature = feature

    def fit(self, y):
        self.model.fit(self.feature, y)

    def predict(self, x):
        return self.predict(x)

    def __str__(self):
        return '%s-%s' % (str(type(self.model)), self.feature)

from automol.features import FeatureGenerator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # noqa
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier # noqa
from sklearn.linear_model import LinearRegression, SGDClassifier # noqa
from sklearn.neural_network import MLPRegressor, MLPClassifier # noqa


def hyperparameter_search(model_name, feature_generator):
    return {
        'RandomForestRegressor': {
            'n_estimators': 42,
        },
    }.get(model_name, {})


class ModelGenerator:

    __modelList = {}
    __modelTypes = {'Regressor': 'regression', 'Regression': 'regression', 'Classifier': 'classification'}

    def __init__(self, feature_generator: FeatureGenerator):
        self.feature_generator = feature_generator
        for gvar, gval in globals().copy().items():
            if issubclass(type(gval), type):
                if any(gvar.endswith(modelType) for modelType in self.__modelTypes):
                    self.__modelList[gvar] = gval

    def get_models(self, problem_type, to_exclude=None):
        return self.generate_models(problem_type, self.__modelList[problem_type].keys() - to_exclude)

    def generate_models(self, problem_type, model_list):
        return [self.generate_model(problem_type, model_name) for model_name in model_list]

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
            # todo
        }[self.get_model_prefix(model_name)]


class ModelAbstraction:
    def __init__(self, model, feature):
        self.model = model
        self.feature = feature

    def fit(self, y):
        self.model.fit(self.feature, y)

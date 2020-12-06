from .features import *


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression


def hyperparameter_search(model_name, feature_generator):
    return {
        'RandomForestRegressor': {
            'n_estimators': 42,
        },
    }.get(model_name, {})


class ModelGenerator:

    __modelList = {
        'RandomForestRegressor': RandomForestRegressor,
        'LinearRegression': LinearRegression,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'GaussianProcessRegressor': GaussianProcessRegressor,
    }

    def __init__(self, feature_generator: FeatureGenerator):
        self.feature_generator = feature_generator

    def generate_models(self, model_list):
        return [self.generate_model(model_name) for model_name in model_list]

    def generate_model(self, model_name):
        if model_name not in ModelGenerator.__modelList:
            raise Exception('unknown model %s' % model_name)

        return ModelGenerator.__modelList[model_name](**hyperparameter_search(model_name, self.feature_generator))

from .features import *


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def hyperparameter_search(model_name, feature_generator):
    return {
        'RandomForestRegressor': {
            'n_estimators': 42,
        },
        'LinearRegression': {

        },
    }[model_name]


class ModelGenerator:

    __modelList = {
        'RandomForestRegressor': RandomForestRegressor,
        'LinearRegression': LinearRegression,
    }

    def __init__(self, feature_generator: FeatureGenerator):
        self.feature_generator = feature_generator

    def generate_model(self, model_name):
        if model_name not in ModelGenerator.__modelList:
            raise Exception('unknown model %s' % model_name)

        return ModelGenerator.__modelList[model_name](**hyperparameter_search(model_name, self.feature_generator))

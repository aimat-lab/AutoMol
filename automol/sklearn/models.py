from __future__ import annotations
import mlflow
import pandas as pd
from automol.models import Model
from sklearn.ensemble import *  # noqa
from sklearn.gaussian_process import *  # noqa
from sklearn.linear_model import *  # noqa
from sklearn.neural_network import *  # noqa
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def hyperparameter_search(model_name):
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

    def generate_all_possible_models(self, problem_type, models_filter=None):
        acceptable_model_names = self.__modelList[problem_type].keys()

        if models_filter is not None:
            acceptable_model_names = acceptable_model_names & set(models_filter['model_names']) \
                if models_filter['whitelist'] \
                else acceptable_model_names - set(models_filter['model_names'])

        return self.generate_models(problem_type, acceptable_model_names)

    def generate_models(self, problem_type, model_names):
        return [SklearnModelGenerator.generate_model(problem_type, model_name)
                for model_name in model_names]

    @staticmethod
    def generate_model(problem_type, model_name):
        type_list = SklearnModelGenerator.__modelList[problem_type]
        if model_name not in type_list:
            raise Exception('unknown model %s' % model_name)
        return SklearnModel(type_list[model_name](**hyperparameter_search(model_name)))


class SklearnModel(Model):

    def __init__(self, core):
        self.core = core
        self.statistics = None

    def run(self, train_features, train_labels, test_features, test_labels):
        mlflow.sklearn.autolog()
        with mlflow.start_run() as run:
            self.core.fit(train_features, train_labels)
            y_pred = self.core.predict(test_features)

            test_mae = mean_absolute_error(test_labels, y_pred)
            test_mse = mean_squared_error(test_labels, y_pred)
            test_r2_score = r2_score(test_labels, y_pred)

            mlflow.log_metric('test_mae', test_mae)
            mlflow.log_metric('test_mse', test_mse)
            mlflow.log_metric('test_r2_score', test_r2_score)
            mlflow.sklearn.log_model(sk_model=self.core, artifact_path='')

            self.statistics = pd.Series(mlflow.get_run(run.info.run_id).data.metrics)

    def get_statistics(self):
        return self.statistics

    def __str__(self):
        return self.core.__str__().replace("()", "")

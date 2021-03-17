from __future__ import annotations
import mlflow
import pandas as pd
from automol.models import Model
from sklearn.ensemble import *  # noqa
from sklearn.gaussian_process import *  # noqa
from sklearn.linear_model import *  # noqa
from sklearn.neural_network import *  # noqa
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np


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
        """
        This method generates all imported sklearn models for a given problem and filter these if necessary.

        Args:
            problem_type: regression or classification
            models_filter: whitelist or blacklist to specify which models to include or exclude respectively

        Returns: list of sklearn models

        """
        acceptable_model_names = self.__modelList[problem_type].keys()

        if models_filter is not None:
            acceptable_model_names = acceptable_model_names & set(models_filter['model_names']) \
                if models_filter['whitelist'] \
                else acceptable_model_names - set(models_filter['model_names'])

        return SklearnModelGenerator.generate_models(problem_type, acceptable_model_names)

    @staticmethod
    def generate_models(problem_type, model_names):
        """
        This method generates all imported sklearn models for a given problem and model names.

        Args:
            problem_type: regression or classification
            model_names: which models to generate

        Returns: list of sklearn models

        """
        return [SklearnModelGenerator.generate_model(problem_type, model_name)
                for model_name in model_names]

    @staticmethod
    def generate_model(problem_type, model_name):
        """
        This method generates a sklearn model for a given problem.

        Args:
            problem_type: regression or classification
            model_name:

        Returns: sklearn model if it is was imported

        """
        type_list = SklearnModelGenerator.__modelList[problem_type]
        if model_name not in type_list:
            raise Exception('unknown model %s' % model_name)
        return SklearnModel(type_list[model_name]())


class SklearnModel(Model):

    def __init__(self, core):
        """
        Wrapper of a sklearn model

        Args:
            core: concrete sklearn model
        """
        self.core = core
        self.statistics = None
        self.param_search = None
        self.learning_curve_data = None

    def run(self, train_features, train_labels, test_features, test_labels, hyper_param_grid, cv, is_learning_curve):
        mlflow.sklearn.autolog()
        with mlflow.start_run() as mlflow_run:
            if hyper_param_grid:
                self.param_search = GridSearchCV(self.core, param_grid=hyper_param_grid,
                                                 cv=cv).fit(train_features, train_labels)
                self.core = self.param_search.best_estimator_
            elif is_learning_curve:
                self.learning_curve_data = learning_curve(self.core, train_features, train_labels,
                                                          train_sizes=np.linspace(0.1, 1.0, 10),
                                                          cv=cv, n_jobs=None, return_times=True)
                return
            else:
                self.core.fit(train_features, train_labels)

            y_pred = self.core.predict(test_features)

            test_mae = mean_absolute_error(test_labels, y_pred)
            test_mse = mean_squared_error(test_labels, y_pred)
            test_r2_score = r2_score(test_labels, y_pred)

            mlflow.log_metric('test_mae', test_mae)
            mlflow.log_metric('test_mse', test_mse)
            mlflow.log_metric('test_r2_score', test_r2_score)
            mlflow.sklearn.log_model(sk_model=self.core, artifact_path='')

            self.statistics = pd.Series(mlflow.get_run(mlflow_run.info.run_id).data.metrics)

    def get_param_search_cv_results(self):
        if self.param_search is not None:
            return pd.DataFrame.from_dict(self.param_search.cv_results_)
        return None

    def get_learning_curve_data(self):
        return self.learning_curve_data

    def get_statistics(self):
        return self.statistics

    def __str__(self):
        return self.core.__str__()

    def get_model_name(self):
        return self.core.__str__().split('(')[0]

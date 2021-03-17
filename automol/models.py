from __future__ import annotations
from typing import List, Dict, Optional, Any


def generate_all_possible_models(problem_type: str,
                                 models_filters: Optional[List[Dict[str, Any]]] = None) -> List[Model]:
    """
    This method generates all possible models for a given problem and filter these if necessary.
    Currently only sklearn models can be generated.

    Args:
        problem_type: regression or classification
        models_filters: whitelist or blacklist to specify which models to include or exclude respectively

    Returns: list of models

    """
    r: List[Model] = []

    for models_filter in models_filters:
        if models_filter['git_uri'] == 'sklearn':
            import automol.sklearn.models
            r += automol.sklearn.models.SklearnModelGenerator().generate_all_possible_models(
                problem_type, models_filter)
        else:
            raise Exception('unknown model type specified %s' % models_filter['type'])

    return r


class Model:

    def run(self, train_features, train_labels, test_features, test_labels, hyper_param_grid, cv, is_learning_curve):
        """
        This method runs a model.

        Args:
            train_features: train data of the feature
            train_labels: train data of the label
            test_features: test data of the feature
            test_labels: test data of the label
            hyper_param_grid: hyper parameters to optimize and possible values
            cv: cross validation parameter, usually defaults to 5
            is_learning_curve: if True then calculates learning curve instead of normal fit and predict routine
        """
        pass

    def get_statistics(self):
        """
        Getter method for statistics with scores like mae, mse
        """
        pass

    def get_model_name(self):
        """
        Getter method for model name without arguments
        """
        pass

    def get_param_search_cv_results(self):
        """
        Getter method for cv scores statistics
        """
        pass

    def get_learning_curve_data(self):
        """
        Getter method for learning curve data
        """
        pass

from __future__ import annotations
from typing import List, Dict, Optional, Any


def generate_all_possible_models(problem_type: str,
                                 models_filters: Optional[List[Dict[str, Any]]] = None) -> List[Model]:
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

    def run(self, train_features, train_labels, test_features, test_labels):
        pass

    def get_statistics(self):
        pass
